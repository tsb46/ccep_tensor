import mne
import numpy as np
import numpy.linalg as la
import os
import pandas as pd 

from glob import glob
from matplotlib import colormaps
from mne.viz import plot_alignment, snapshot_brain_montage
from patsy import dmatrix
from sklearn.linear_model import Ridge


class DataLoad:
    """
    Load and concatenate CCEP data from multiple runs from a single
    subject
    
    Attributes
    ----------
    subject: str
        the subject label (according to participants.tsv file).
    session: str
        the session label (according to participants.tsv file)
    data_dir: str
        path to data directory (default: 'data/ccep')
    fsaverage_dir:
        path to directory containing fsaverage directory (default: 'data')
    
    Methods
    -------
    load(fp)
        load surface projection files and stack
    """
    def __init__(self, subject, session, data_dir='data/ccep', 
                 fsaverage_dir='data'):
        self.subject = subject
        self.session = session
        self.data_dir = data_dir
        self.fsaverage_dir = fsaverage_dir
        # check files are in data dir
        self.epo_fps = self._find_epo_files() 
        # check electrode metadata files are in data dir
        self.el_df_fp, self.el_dist_fp = self._find_el_files()

    def _find_el_files(self):
        # find electrode metadata files 
        el_meta_fp = f'{self.data_dir}/{self.subject}_{self.session}_electrodes_epochs.csv'
        if not os.path.isfile(el_meta_fp):
            raise Exception(
                f'no electrode metadata file found: {el_meta_fp}'
            )
        el_dist_fp = f'{self.data_dir}/{self.subject}_{self.session}_electrodes_dist_epochs.csv'
        if not os.path.isfile(el_dist_fp):
            raise Exception(
                f'no electrode distance file found: {el_dist_fp}'
            )
        return el_meta_fp, el_dist_fp

    def _find_epo_files(self):
        # find cceps for all runs
        subj_fps = glob(f'{self.data_dir}/{self.subject}*_epo.fif*')
        if len(subj_fps) < 1:
            raise Exception(
                f'no epoch files found in {self.data_dir} for subject {self.subject}'
            )
        subj_fps = sorted(subj_fps)
        return subj_fps

    def _load_epochs(self, fp):
        ccep_data = mne.read_epochs(fp, verbose=False)
        return ccep_data        

    def load(self):
        # loop through runs and load
        ccep_epochs  = []
        for i, epo_fp in enumerate(self.epo_fps):
            ccep = self._load_epochs(epo_fp)
            ccep_epochs.append(ccep)
            if i == 0:
                self.montage = ccep.get_montage()
        # ensure channels are aligned across epochs
        ccep_epochs = mne.equalize_channels(ccep_epochs, verbose=False)
        # concatenate into one epoch object
        ccep_epochs = mne.concatenate_epochs(ccep_epochs, verbose=False)
        self.ch_names = ccep_epochs.ch_names
        # load electrode metadata
        el_df = pd.read_csv(self.el_df_fp)
        el_dist = pd.read_csv(self.el_dist_fp)
        return ccep_epochs, el_df, el_dist
        

class LagSpline:
    """
    Create basis along lags of stimulation impulse using natural cubic
    regression splines. Specifically, the spline basis is represented across time lags
    of the time course. It is used to weight columns of a lagged time course where lagged (1,..,N) 
    copies of the original time course are appended as columns from the left to the right. 

    This approach allows us to flexibly fit the ccep across 
    time (from onset to an arbitrary time length). The number of splines controls 
    how flexible we allow the predicted evoked ccep waveform to be. With more splines, 
    we allow more wiggly waveforms. 

    Attributes
    ----------
    times: np.array
        numpy array of epoch time samples indexed to the onset (t=0)
    sf: float
        sampling frequency of data
    window: list
        window (ms) post-stimulation onset to model the CCEP waveform specified as a list
        with start time as the first entry and end time as the second entry. If None is specified
        for the first entry, the stimulation onset time is used (0.0). This is not recommended to 
        avoid the stimulation artifact. The preprocessing pipeline interpolates this artifact, but
        not a good idea to use the interpolated data either. If None is specified for the second 
        entry, the last time point of the epoch is used (Default: (0.008, None)). 
    start_knot = float
        the placement of the first knot (in ms) from stim onset. This should be carefully placed to 
        pick up the rapid N1 response waveform (Default: 0.015)
    end_knot = float
        the placement of the end knot (in ms) from stim onset. This should be set before the time set
        as the end of the window. Ideally, it should be set around the point you expect the waveform 
        to end. N2 peak latency is 50-300ms. By default it's set at 400 ms (Default: 0.4).
    n_knots: int
        number of knots for splines to model evoked CCEP waveforms with (default: 6).
    onset_time: float
        onset of stimulation in CCEP trial. MNE CCEP object sets this as 0 (Default = 0.0)
    
    Methods
    -------
    construct_basis(event_ts)
        project lagged time course of stimulation impluse into spline basis 
    basis_project(n_eval)

    """
    def __init__(self, times, sf, window=[0.008, None], start_knot = 0.01,
                 end_knot=0.4, n_knots=6, onset_time=0.):
        self.n_knots = n_knots
        self.times = times
        self.sf = sf
        self.window = window
        self.start_knot = start_knot
        self.end_knot = end_knot
        self.onset_time = onset_time

        # if first entry in window is None, set to stim onset (0.0)
        if self.window[0] is None:
            self.window[0] = 0
        # if last entry in window is None, set to end of epoch
        if window[1] is None:
            # epoch times are centered at 0 so take last time point
            self.window[1] = times[-1]

        # get window indices from time vector
        self.win_start_i = np.argmin(np.abs(times - window[0]))
        self.win_end_i = np.argmin(np.abs(times - window[1]))
        self.win_len = self.win_end_i - self.win_start_i
        # create a mask to select window time points
        self.win_mask = np.zeros(len(times)) 
        self.win_mask[self.win_start_i:(self.win_end_i+1)] = 1
        self.win_mask = self.win_mask.astype(bool)
        # check end knot location is before end of window
        if self.end_knot >= self.window[1]:
            raise Exception(
                f'end knot placement {self.end_knot} is after '
                f'end of window {self.window[1]}'
            )

        # get start and end knot indices relative to stim onset index
        start_knot_i = np.argmin(np.abs(times - self.start_knot))
        end_knot_i = np.argmin(np.abs(times - self.end_knot))
        self.start_knot_i = start_knot_i - self.win_start_i
        self.end_knot_i =  end_knot_i - self.win_start_i

    def _create_event_impulse(self):
        # create event time course with single impulse
        onset_idx = self.win_start_i
        event_ts = pd.Series(np.zeros(len(self.times)), index=self.times)
        event_ts.iloc[onset_idx] = 1
        return event_ts

    def basis_project(self, n_eval, eval_val=1):
        # project lags of a single scalar predictor value into basis
        # this is for evaluating the predictions of a model
        eval_basis = np.linspace(0, self.lag_vec[-1], n_eval)
        eval_time = self.window[0] + eval_basis/self.sf
        basis_pred = dmatrix(self.basis.design_info, {'x': eval_basis}, 
                         return_type='matrix')
        # Intialize basis matrix
        pred_list = [eval_val * basis_pred[:, l] 
                     for l in range(basis_pred.shape[1])]
        basis_pred = np.vstack(pred_list).T
        return eval_time, basis_pred

    def construct_basis(self):
        # construct basis over lags of event impulse
        # create stimulation impulse vector
        event_ts = self._create_event_impulse()
        # arange knot placement w/ geometric spacing w/ more emphasis on early time points
        self.lag_vec = np.arange(self.win_len).astype(int)
        self.knots = np.geomspace(self.start_knot_i, self.end_knot_i, 
                                  self.n_knots)
        # create basis
        self.basis = dmatrix("cr(x, knots=self.knots) - 1", 
                        {"x": self.lag_vec}, return_type='matrix')
        # put event 'impulse' in basis
        event_basis = np.zeros((len(event_ts), self.basis.shape[1]))
        # lag event time course 'n' times as specified in 'lag_vec'
        lag_mat = np.vstack(
            [event_ts.shift(l, fill_value=0).values for l in self.lag_vec] 
        ).T
        # Loop through splines and multiply with lagged event time courses
        for l in np.arange(self.basis.shape[1]):
            event_basis[:, l] = np.dot(lag_mat, self.basis[:,l])
        return event_basis


class CCEPModel(DataLoad):
    """
    Perform regression modeling of CCEP response to pre-stimulus oscillatory
    power with covariates. A flexible spline basis is used to model the time-varying
    CCEP waveform conditioned on prestimulus power.
    ...

    Attributes
    ----------
    subject: str
        the subject label (according to participants.tsv file).
    session: str
        the session label (according to participants.tsv file)
    data_dir: str
        path to data directory (default: 'data/ccep')
    window: list
        window (ms) post-stimulation onset to model the CCEP waveform specified as a list
        with start time as the first entry and end time as the second entry. If None is specified
        for the first entry, the stimulation onset time is used (0.0). This is not recommended to 
        avoid the stimulation artifact. The preprocessing pipeline interpolates this artifact, but
        not a good idea to use the interpolated data either. If None is specified for the last time
        point of the epoch is used (Default: (0.008, None)). 
    norm: bool
        whether to z-score normalize concatenated epoch time courses (Default: True)
    start_knot = float
        the placement of the first knot (in ms) from stim onset. This should be carefully placed to 
        pick up the rapid N1 response waveform (Default: 0.01)
    end_knot = float
        the placement of the end knot (in ms) from stim onset. This should be set before the time set
        as the end of the window. Ideally, it should be set around the point you expect the waveform 
        to end. N2 peak latency is 50-300ms. By default it's set at 400 ms (Default: 0.4).
    n_knots: int
        number of knots for splines to model evoked CCEP waveforms with (default: 6).
    min_trial: int
        minimum number of trials each stimulation pair (mapped to Destrieux labels)
        must have to be included as separate factor in the model. Stimulation pair
        epochs that do not meet threshold are dropped (Default: 10).

    Methods
    -------
    run()
        load subject ccep data, concatenate across runs, and perform
        regression modeling
    predict(stim_label, ch_pick=None, n_eval=400)
        predict CCEP waveform in response to stimulation (within) a 
        particular Destrieux ROI

    """

    def __init__(self, subject, session, data_dir='data/ccep',
                 window=[0.009, None], norm=True, start_knot = 0.01,
                 end_knot=0.4, n_knots=6, min_trial=10,
                 verbose=True):
        super().__init__(subject=subject, session=session, data_dir=data_dir)
        self.window = window
        self.norm = norm
        self.start_knot = start_knot
        self.end_knot = end_knot
        self.n_knots = n_knots
        self.min_trial = min_trial
        self.verbose = verbose

    def _ch_stim_dummy(self, stim_labels):
        # create dummy-coded stim channel predictors
        # set most frequent stim epoch as reference
        stim_ref = stim_labels.mode()[0]
        stim_dmatrix = dmatrix(
            f"C(x, Treatment('{stim_ref}')) - 1", 
            {"x": stim_labels}, return_type='matrix'
        )
        # keep in object for evaluating predictions
        self.stim_dmatrix = stim_dmatrix
        # keep different stim conditions in object
        self.stim_labels = stim_labels.unique()
        return stim_dmatrix

    def create_design_mat(self):
        # create design matrix for regression
        # create Pandas dataframe of CCEP waveforms
        ccep_df = self._extract_epoch_df()
        # create spline basis from lags of stimulation time course
        self.basis = LagSpline(
            self.ccep_epochs[0].times, 
            self.ccep_epochs.info['sfreq'], 
            window=self.window, 
            n_knots=self.n_knots,
            start_knot = self.start_knot,
            end_knot=self.end_knot
        )
        stim_basis = self.basis.construct_basis()
        # repeat stim basis for each epoch
        stim_basis = np.vstack([stim_basis]*len(self.ccep_epochs))
        # repeat window mask for each epoch
        self.ccep_mask = np.tile(self.basis.win_mask, len(self.ccep_epochs))
        # dummy code epoch labels
        stim_dmatrix = self._ch_stim_dummy(ccep_df['condition_d'])
        # create pairwise interactions between stimulation basis
        # and epoch dummies to allow CCEP waveform to vary by location
        # of stimulus
        design_mat = self.inter_pairwise(stim_basis, stim_dmatrix)
        # pull channel time courses from ccep dataframe
        ccep_df_ch = ccep_df[self.ch_names] 
        return design_mat, ccep_df_ch

    def _extract_epoch_df(self):
        # convert ccep data into Pandas dataframe
        ccep_df = self.ccep_epochs.to_data_frame()
        # get metadata
        ccep_meta = self.ccep_epochs.metadata.copy()
        # reset index to ensure index is sequentially increasing 
        # (needed to match epoch indx from ccep object)
        ccep_meta = ccep_meta.reset_index(drop=True)
        # remove duplicate epochs 
        epoch_meta = ccep_meta.groupby('epoch').first().reset_index()
        # merge epoch metadata with ccep df
        epoch_meta = epoch_meta.rename(columns = {'epoch': 'condition'})
        ccep_df = ccep_df.merge(epoch_meta, on='condition', how='left')
        # create stimulation pairs based on Destrieux label
        label_cols = ['Destrieux_label_text_stim_pair_1', 
                      'Destrieux_label_text_stim_pair_2']
        ccep_df['condition_d'] = \
        ccep_df[label_cols[0]] + '__' + ccep_df[label_cols[1]]
        # drop epochs that do not minimum # of trials threshold
        # filter to one time point to deduplicate epoch column
        mask = ccep_df.time == ccep_df.time.min()
        # count # of trials per epoch
        trial_n = ccep_df.loc[mask]['condition_d'].value_counts()
        # find epochs that do not meet min # of trials 
        epoch_drop = trial_n[trial_n < self.min_trial].index.tolist()
        # split epoch label into two stim pairs
        epoch_drop = [e for e_pair in epoch_drop for e in e_pair.split('__')]
        # create mask for metadata df
        mask_drop_meta = ccep_meta[label_cols[0]].isin(epoch_drop) | \
        ccep_meta[label_cols[1]].isin(epoch_drop)
        # get indices of epochs without sufficient trials
        epoch_drop_idx = ccep_meta.loc[mask_drop_meta].index
        self.ccep_epochs.drop(epoch_drop_idx)
        # create mask for ccep_df
        mask_drop_ccep = ccep_df[label_cols[0]].isin(epoch_drop) | \
        ccep_df[label_cols[1]].isin(epoch_drop)
        # filter out epochs from design mat without sufficient trials
        ccep_df = ccep_df.loc[~mask_drop_ccep].copy()
        return ccep_df

    @staticmethod
    def inter_pairwise(A,B):
        # construct pairwise interaction b/w cols of two dataframes
        # then append to 'main effects dataframes'
        a_ncols = A.shape[1]
        b_ncols = B.shape[1]
        AB = np.vstack(
            [A[:,a]*B[:, b] for a in range(a_ncols) for b in range(b_ncols)]
        ).T
        A_B_AB = np.hstack([A,B,AB])
        return A_B_AB

    def predict(self, stim_label, ch_pick=None, n_eval=400):
        self._validate_fit()
        # predict CCEP waveform in response to stimulation within a given Destrieux label
        # get stimulation basis            
        time, basis_pred = self.basis.basis_project(n_eval)
        # get stim dummy dataframe
        pred_stim_dmat = dmatrix(
            self.stim_dmatrix.design_info, {'x': [stim_label]}, 
            return_type='matrix'
        )
        # repeat stim_dummy rows to match stimulation basis
        pred_stim_dmat = np.vstack([pred_stim_dmat]*basis_pred.shape[0])
        pred_design_mat = self.inter_pairwise(basis_pred, pred_stim_dmat)
        if ch_pick is not None:
            ch_idx = self.ccep_ts.columns.get_loc(ch_pick)
            ccep_pred = self.reg[ch_idx].predict(pred_design_mat)
        else:
            ccep_pred = []
            for ch in self.ch_names:
                ch_idx =  self.ccep_ts.columns.get_loc(ch)
                pred_ts = self.reg[ch_idx].predict(pred_design_mat)
                ccep_pred.append(pred_ts)
        return time, ccep_pred

    def regression(self, X, Y):
        """
        Regress design matrix (X) onto CCEP time courses (Y) using
        Ridge regression for more stable estimates. Due to the missing 
        values in each column we can't fit in one pass. Loop through 
        individual time courses, mask missing and fit separate models.
        """
        self.reg = []
        self.reg_coef = []
        self.reg_r2_full = []
        self.reg_r2_win = []
        # lop through channels and fit regression model
        for ch in Y.columns:
            # index channel time course
            y = Y[ch].copy()
            y_mask = ~y.isna()
            y_win_mask = ~y.isna() & self.ccep_mask
            # if if specified, normalize time course before fit
            if self.norm:
                y_mean = y[y_mask].mean()
                y_std = y[y_mask].std()
                y = (y[y_mask] - y_mean)/y_std
                y_win = (y[y_win_mask] - y_mean)/y_std
            else:
                y = y[y_mask]
                y_win = y[y_win_mask]
            # initiilize ridge regression model
            ridge = Ridge()
            ridge.fit(X[y_mask,:], y)
            self.reg.append(ridge)
            self.reg_coef.append(ridge.coef_)
            self.reg_r2_full.append(ridge.score(X[y_mask,:], y))
            self.reg_r2_win.append(
                ridge.score(X[y_win_mask,:], y_win)
            )

    def run(self):
        # run CCEP regression analysis
        # load data
        self.ccep_epochs, self.el_df, self.el_dist = super().load()
        # create design matrix
        design_mat, self.ccep_ts = self.create_design_mat()
        # fit regression model
        self.regression(design_mat, self.ccep_ts)

    def _validate_fit(self):
        if not hasattr(self, 'reg'):
            raise Exception('must fit model with .run() before visualization')

    def visualize_ccep(self, stim_label, hemi, n_eval=400):
        self._validate_fit()
        time, ccep_pred = self.predict(stim_label, n_eval=n_eval)
        ccep_pred = np.vstack(ccep_pred)
        evoked = mne.EvokedArray(ccep_pred, self.ccep_epochs.info, 
                                 tmin=time[0])
        evoked.set_montage(self.montage)
        # load fsaverage source space
        src = mne.read_source_spaces(
            "data/fsaverage/bem/fsaverage-ico-5-src.fif"
        )
        # project component to surface space
        stc = mne.stc_near_sensors(
            evoked,
            trans="fsaverage",
            subject="fsaverage",
            subjects_dir=self.fsaverage_dir,
            src=src,
            surface="pial",
            mode="sum",
            distance=0.02,
        )
        # plot surface projectrion on fsaverage pial surface
        e_std = evoked.data.std()
        clim_vals = [0, e_std, e_std*2]
        clim = dict(
            kind="value", 
            pos_lims=clim_vals
        )
        brain = stc.plot(
            surface="pial",
            hemi=hemi,
            colormap="coolwarm",
            colorbar=False,
            clim=clim,
            title=stim_label,
            views=["lat", "med"],
            subjects_dir=self.fsaverage_dir,
            size=(500, 500),
            smoothing_steps=5,
            time_viewer=True,
        )

    def visualize_electrodes(self):
        # plot electode locations on pial surface
        self._validate_fit()
        fig = plot_alignment(
            self.ccep_epochs.info,
            trans="fsaverage",
            subject="fsaverage",
            subjects_dir=self.fsaverage_dir,
            surfaces=["pial"],
            coord_frame="head",
        )
        mne.viz.set_3d_view(fig, azimuth=0, elevation=70)
        xy, im = snapshot_brain_montage(fig, self.ccep_epochs.info)







    
    