import boto3
import json
import mne
import numpy as np
import os
import pandas as pd

from .utils import run_once
from botocore import UNSIGNED
from botocore.client import Config
from concurrent import futures
from concurrent.futures import ProcessPoolExecutor
from glob import glob
from itertools import repeat
from mne_bids import (
    BIDSPath, get_entities_from_fname, 
    read_raw_bids
)
from mne.baseline import rescale
from mne.datasets import fetch_fsaverage
import pandas as pd
from scipy.spatial import KDTree
from scipy.spatial.distance import pdist, squareform

# set up S3 client
boto3.setup_default_session()
s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))


class Downloader:
    """
    Download subject files from OpenNeuro dataset via AWS
    ...

    Attributes
    ----------
    subjects: list
        the subject labels (according to participants.tsv file)
    session: str
        the session labels (according to participants.tsv file)
    out_dir: str
        path to directory for writing files
    max_workers: int
        maximum number of cores for downloading files in parallel (default = None)

    Methods
    -------
    download(include=None)
        downloads files in subject directory
    """
    def __init__(self, session, subjects, out_dir, 
                 max_workers=None):
        self.subjects = subjects
        self.session = session
        self.out_dir = out_dir
        self.bucket = 'openneuro.org'
        self.dataset = 'ds004080'
        self.task = 'SPESclin'
        self.max_workers = None
        # get dataset sub-directory
        self.prefix_base = self._find_dataset()

    @run_once
    def download(self, include=None):
        """
        downloads files from subjects to out_dir with 
        inclusion criteria.
        
        Parameters
        ----------
        include: str, optional
            a wildcard expression for including specific file types (e.g. '*.nii')

        Returns
        -------
        None
        """
        # loop through subjects
        self.subject_meta = {} # collect subject and run labels
        for s, ses in zip(self.subjects, self.session):
            # get subject files 
            file_keys = self._find_files(s, ses, include)
            # get runs
            runs = self._get_runs(file_keys)
            # iterate through files (in parallel) and download
            download_dir = self._get_download_path_bids(s, ses)
            download_iter = (file_keys, download_dir)
            out_files = []
            for key, result, out in self._download_parallel_multiprocessing(*download_iter):
                print(f"{key}: {result}")
                out_files.append(out)
            # collect metadata
            self.subject_meta[s] = {
                'fps': out_files,
                'dir': download_dir,
                'runs':  runs
            }

        # set download flag to True
        self.download_flag = True


    def _download_object(self, file_path, download_dir):
        # https://www.learnaws.org/2022/10/12/boto3-download-multiple-files-s3/
        """Downloads an object from S3 to local."""

        s3_client = boto3.client("s3")
        os.makedirs(download_dir, exist_ok=True)
        download_path = os.path.join(download_dir, os.path.basename(file_path))
        print(f"Downloading {file_path} to {download_path}")
        s3.download_file(
            self.bucket,
            file_path,
            str(download_path)
        )
        return "Success", download_path

    def _download_parallel_multiprocessing(self, file_keys, download_dir):
        # https://www.learnaws.org/2022/10/12/boto3-download-multiple-files-s3/
        with ProcessPoolExecutor(self.max_workers) as executor:
            future_to_key = {
                executor.submit(self._download_object, key, download_dir): key 
                for key in file_keys
            }

            for future in futures.as_completed(future_to_key):
                key = future_to_key[future]
                exception = future.exception()

                if not exception:
                    res = future.result()
                    yield key, res[0], res[1]
                else:
                    yield key, exception

    def _find_dataset(self):
        """
        find dataset bucket, raise error if nothing (or multiple) found
        """
        # get dataset bucket
        dataset_dir = s3.list_objects_v2(
            Bucket=self.bucket, Prefix=self.dataset, Delimiter='/'
        )
        if len(dataset_dir['CommonPrefixes']) > 1:
            raise Exception(f'multiple datasets found: {dataset_dir["CommonPrefixes"]}')
        elif len(dataset_dir['CommonPrefixes']) < 1:
            raise Exception(f'{self.dataset} not found in OpenNeuro bucket')
        else:
            dataset_path = dataset_dir['CommonPrefixes'][0]['Prefix']
        return dataset_path

    def _find_files(self, subject, session, include):
        """
        find files within subject directory
        """
        # list files
        objs = s3.list_objects_v2(
            Bucket='openneuro.org', 
            Prefix=f'{self.prefix_base}{subject}/{session}/ieeg/'
        )
        file_keys = [obj['Key'] for obj in objs['Contents']]
        return file_keys

    def _get_download_path_bids(self, subject, session):
        """
        define the download path in BIDS format
        """
        d_path = os.path.abspath(
            f'{self.out_dir}/{subject}/{session}/ieeg'
        )
        return d_path

    def _get_runs(self, file_keys):
        """
        find separate runs in subject directory
        """
        f_eeg = [f for f in file_keys if os.path.splitext(f)[1] == '.eeg']
        runs = list(set([get_entities_from_fname(f)['run'] for f in f_eeg]))
        return runs


class Pipeline(Downloader):
    """
    Preprocessing pipeline to download subject eeg files and preprocesses 
    data file from raw to CCEPs
    ...

    Attributes
    ----------
    subjects: list
        the subject labels (according to participants.tsv file)
    session: str
        the session labels (according to participants.tsv file)
    out_dir: str
        path to directory to download files to (or where they are currently stored)
        (default: 'data')
    preproc_dir: str
        path to directory to write preprocessed files to
        (default: '../data/preproc')
    epoch_range: tuple
        time range of epoch in secs (tmin, tmax) (default: (-2, 1))
    baseline_range: tuple
        time range of baseline in secs in reference 
        to start of epoch (t = 0). Must be contained within
        epoch range. (default =  (-2, -0.1) ). Baseline correction
        is performed via z-scoring.
    interp_range: tuple
        time range (sec) to interpolate (cubic interpolation) to remove 
        stimulation artefact relative to onset (default = (-0.008, 0.008)
    resample: float
        frequency to resample epoched data (default: 400Hz)
    amp_reject: float
        min-max amplitude threshold in z-score units for rejecting epochs (default: 10.0)
    max_workers: int
        maximum number of cores for downloading files in parallel (default = None)
    stim_el_dist: float
        maximum distance (mm) from stimulating electrodes for which to ignore electrode
        time courses (default: 13.0)

    Methods
    -------
    download(include=None)
        downloads files in subject directory (from Downloader class)
    preprocess(eeg, events_df)
        preprocess eeg from raw to epoched data
    """
    def __init__(self, subjects, session, download_exists=False, out_dir='data', 
                 epoch_range=(-2.0, 1.0), baseline_range=(-2.0, -0.1), 
                 interp_range=(-0.002, 0.008), resample=400, amp_reject=10.0,
                 max_workers=None, stim_el_dist=13.0, verbose=True):
        super().__init__(subjects=subjects, session=session, 
                         out_dir=out_dir, max_workers=max_workers)
        self.epoch_range = epoch_range
        self.baseline_range = baseline_range
        self.interp_range = interp_range
        self.resample = resample
        self.amp_reject = amp_reject
        self.stim_el_dist = stim_el_dist
        self.verbose = verbose

        # create directory for preprocessed files
        self.out_dir_p = f'{self.out_dir}/ccep'
        os.makedirs(self.out_dir_p, exist_ok=True)
        # if download_exists flag passed, don't download again
        if download_exists:
            if verbose:
                print('finding downloaded files for subjects')
            # iterate subjects and pull metadata on files
            self.subject_meta = {}
            for s, ses in zip(self.subjects, self.session):
                download_dir = self._get_download_path_bids(s, ses)
                fps, runs = self.find_downloaded_files(download_dir, s)
                self.subject_meta[s] = {
                    'fps': fps,
                    'dir': download_dir,
                    'runs':  runs
                }
            self.download_flag = True

    def _annotate(self, raw, events_df):
        """
        annotate data with modified event timing
        """
        # get onsets
        onset = events_df.onset
        # get duration
        dur = events_df.duration
        # get 'description'
        e_stim = events_df.electrical_stimulation_site
        e_trial = events_df.trial_type
        # rename 'artefact' as 'bad' so MNE ignores those epochs
        desc = ['bad' if t == 'artefact' else e for e, t in zip(e_stim, e_trial)]
        # set new annotations in raw MNE object
        stim_annot = mne.Annotations(
            onset=onset,  # in seconds
            duration=dur,  # in seconds, too
            description=desc,
        )

        raw.set_annotations(stim_annot)
        return raw

    def _create_metadata(self, epoch, stim_labels):
        # create dataframe of metadata to specify covariates for
        # each epoch
        epoch.metadata = pd.DataFrame(
            stim_labels, index=range(len(epoch)),
            columns=['epoch']
        )
        # split out each stim channel in stimulation epoch
        epoch_split = epoch.metadata.epoch.str.split('-')
        epoch.metadata['stim_pair_1'] = epoch_split.str[0].str.strip().str.lower()
        epoch.metadata['stim_pair_2'] = epoch_split.str[0].str.strip().str.lower()
        # merge stim Destriex-label
        destrieux_label = ['name', 'Destrieux_label', 'Destrieux_label_text']
        for ch in ['stim_pair_1', 'stim_pair_2']:
            epoch.metadata = epoch.metadata.merge(
                self.el_df[destrieux_label], left_on=ch,
                right_on='name', how='left'
            )
            epoch.metadata = epoch.metadata.drop(columns=['name'])
            col_rename = {
                'Destrieux_label_text': f'Destrieux_label_text_{ch}',
                'Destrieux_label': f'Destrieux_label_{ch}'
            }
            epoch.metadata = epoch.metadata.rename(columns=col_rename)
        return epoch

    def _el_dist_nn(self, ch_names):
        el_df_ch = self.el_df.loc[self.el_df.name.isin(ch_names)]
        el_coord = el_df_ch[['x', 'y', 'z']].values
        el_kdtree = KDTree(el_coord, leafsize=10)
        return el_kdtree

    def _epoch(self, raw, bad_channels):
        # create epoched mne object from annotations and raw mne object

        # create custom event to description mapping to make 
        # the same stimulation trials consistent across runs
        annot_labels = np.unique(raw.annotations.description)
        # filter out 'bad' (artifact) events
        annot_labels = [l for l in annot_labels if l != 'bad']
        if self.run_idx == 0:
            self.event_dict = {l: i for i, l in enumerate(annot_labels)}
        else:
            # add any new stimulation trials as separate events
            max_idx = max(self.event_dict.values())
            annot_labels = [l for l in annot_labels if l not in self.event_dict]
            for i, l in enumerate(annot_labels, max_idx+1):
                self.event_dict[l] = i

        events_from_annot, event_dict = mne.events_from_annotations(
            raw, event_id=self.event_dict
        )
        epochs = mne.Epochs(raw, events_from_annot, tmin=self.epoch_range[0], 
                            tmax=self.epoch_range[1], event_id=event_dict, 
                            preload=True, baseline=None)
        
        # drop 'nan' events
        id_to_event = {i: e for e,i in epochs.event_id.items()}
        # lowercase all channels and bad channel labels for matching
        bad_channels = [c.lower() for c in bad_channels]
        ch_names = [c.lower() for c in epochs.ch_names]
        # remove 'nan' trials and trials w/ 'bad' channels as stim channels
        bad_events = []
        for i, e in enumerate(epochs.events):
            e_label = id_to_event[e[2]]
            if e_label == 'nan':
                bad_events.append(i)
                continue
            e_stim1, e_stim2 = [e.strip().lower() for e in e_label.split('-')]
            if (e_stim1 in bad_channels) or (e_stim2 in bad_channels):
                bad_events.append(i)
                continue
            # remove stim channels that do not exist in channel names
            if (e_stim1 not in ch_names) or (e_stim2 not in ch_names):
                bad_events.append(i)
        # drop identified trials
        epochs.drop(bad_events)

        return epochs

    def _load_eeg(self, subject, session, run):
        # load brainvision file into MNE
        subj_str = subject.replace('sub-', '')
        ses_str = session.replace('ses-', '')
        bids_path = BIDSPath(
            root=self.out_dir, 
            subject=subj_str,
            session=ses_str,
            task='SPESclin',
            run=run,
            suffix='ieeg'
         )
        raw = read_raw_bids(bids_path=bids_path, verbose='error')
        return raw

    def _load_event(self, subj, session, run):
        # load event file in BIDS format
        subject_dir = f'{self.out_dir}/{subj}/{session}/ieeg'
        event_fp = os.path.abspath(
            f'{subject_dir}/{subj}_{session}_task-{self.task}_run-{run}_events.tsv'
        )
        events_df = pd.read_csv(event_fp, delimiter='\t')
        # sort events by onset
        events_df = events_df.sort_values(by='onset')
        # remove stimulation 'event'
        events_df = events_df.loc[events_df.trial_type != 'stimulation'].copy()
        return events_df

    def _load_electrodes(self, subj, session):
        # load electrode file and process for further analysis
        subject_dir = f'{self.out_dir}/{subj}/{session}/ieeg'
        el_fp = os.path.abspath(
            f'{subject_dir}/{subj}_{session}_electrodes.tsv'
        )
        el_df = pd.read_csv(el_fp, delimiter='\t')
        # drop non-ecog electrode rows
        el_df = el_df.loc[el_df.group.isin(['strip', 'grid'])].copy()
        # select relevant columns
        el_cols = ['name', 'x', 'y', 'z', 'Destrieux_label', 
                   'Destrieux_label_text']
        el_df = el_df[el_cols].copy()
        # lowercase and strip 'name' column for matching later
        el_df['name'] = el_df.name.str.strip().str.lower()
        # create a 'long' dataframe of euclidean distances b/w each pair of electrodes
        el_dist = pd.DataFrame(
            squareform(pdist(el_df[['x','y','z']].values)),
            index=el_df['name'].values,
            columns=el_df['name'].values
        ).unstack().reset_index()
        el_dist.columns = ['channel_1', 'channel_2', 'dist']
        # write out electrode and electrode distance files for future analysis
        el_fp_out = os.path.abspath(
            f'{self.out_dir_p}/{subj}_{session}_electrodes_epochs.csv'
        )
        el_dist_fp_out = os.path.abspath(
            f'{self.out_dir_p}/{subj}_{session}_electrodes_dist_epochs.csv'
        )
        el_df.to_csv(el_fp_out)
        el_dist.to_csv(el_dist_fp_out)
        return el_df, el_dist

    def find_downloaded_files(self, download_dir, subject):
        # find files and runs for each subject
        subj_fps = glob(f'{download_dir}/*')
        if len(subj_fps) < 1:
            raise Exception(f'no files found in {download_dir} for subject {subject}')
        runs = sorted(self._get_runs(subj_fps))
        return subj_fps, runs

    def _fill_nan(self, x, times, start, end):
        # fill span with NaNs
        start_idx = np.argmin(np.abs(times - start))
        end_idx = np.argmin(np.abs(times - end))
        x[(start_idx-1):(end_idx+1), :] = np.NaN
        return x

    def _interpolate_stim_artifact(self, X, times):
        # iterate through epochs and interpolate time around stimulation artifact
        # with cubic interpolation using Pandas interpolation function
        for i in range(X.shape[0]):
            x = np.squeeze(X[i, :, :]).T
            x = self._fill_nan(x, times, self.interp_range[0], self.interp_range[1])
            #  Piecewise Cubic Hermite Interpolating Polynomial to enforce monotonicity
            x_interp = self._pd_interp_nan(x, times, 'pchip')
            X[i, :, :] = x_interp.T
        return X

    def _nan_channel(self, X, ch_labels, stim_labels, el_kdtree):
        # iterate through epochs and replace stim and nearby channels with NaN
        for i in range(X.shape[0]):
            # get labels of stim channels
            ch_stim = stim_labels[i]
            ch_stim_pair = ch_stim.split('-')
            # get stim indices
            ch_stim_indx = [ch_labels.index(c.strip().lower()) 
                            for c in ch_stim_pair]
            # get stim coordinates
            ch_stim_coord = [el_kdtree.data[i,:] for i in ch_stim_indx]
            # query tree for electrodes nearby (default 13mm) stim electrodes
            nearby_el_1 = el_kdtree.query_ball_point(ch_stim_coord[0], self.stim_el_dist)
            nearby_el_2 = el_kdtree.query_ball_point(ch_stim_coord[1], self.stim_el_dist)
            nearby_el = nearby_el_1 + nearby_el_2
            # set channels to NaN
            ch_interp = [ch_labels[c_i] for c_i in nearby_el]
            ch_interp_bool = np.array([c in ch_interp for c in ch_labels])
            X[i,ch_interp_bool,:] = np.nan
            # find channels that exceed amplitude threshold and nan
            amp_reject = (np.abs(X[i,:,:]) > self.amp_reject).any(axis=1)
            X[i, amp_reject, :] = np.nan
        return X

    def _pd_interp_nan(self, x, times, method, channel=True):
        # interpolate time courses via Pandas interpolate method
        x = pd.DataFrame(x, index=times)
        x_interp = x.interpolate(method=method)
        return x_interp

    @run_once
    def preprocess(self):
        """
        load and preprocess raw mne object: epoch, annotate,
        interpolate stimulation channels, and resample
        
        Parameters
        ----------
        raw: mne.io.Raw
            raw MNE object
        event_df: pd.DataFrame
            event timing in Pandas dataframe

        Returns
        -------
        None
        """
        # check files have been downloaded
        if not hasattr(self, "download_flag"):
            raise Exception(
                """
                dataset must be downloaded from AWS - use self.download()
                or pass download_exists=True (if already downloaded)
                """
            )
        ses, task = self.session, self.task
        # create dict to record metdata
        # loop through subjects
        for s, ses in zip(self.subjects, self.session):
            # load electrode metadata
            self.el_df, self.el_dist = self._load_electrodes(s, ses)
            # set run index to keep up with runs per subject
            self.run_idx = 0
            for r in self.subject_meta[s]['runs']:
                if self.verbose:
                    print(f'preprocessing subject {s} - run {r}')
                # load eeg
                raw = self._load_eeg(s, ses, r)
                # load event time file into pandas df
                events_df = self._load_event(s, ses, r)
                # apply preprocessing pipeline
                epoch = self._run_pre_pipe(raw, events_df)
                # write out ccep data for each run
                out_dir = f'{self.out_dir_p}/'
                out_fp = f'{s}_ses-{ses}_task-{task}_run-{r}_epo.fif.gz'
                out_fp = os.path.abspath(os.path.join(out_dir, out_fp))
                epoch.save(out_fp, overwrite=True)
                # increase run index
                self.run_idx += 1

    def _run_pre_pipe(self, raw, events_df):
        """
        run instance of preprocessing on single IEEG dataset 
        """
        # annotate dataset with modified event timing
        raw = self._annotate(raw, events_df)
        # pick ecog channels
        raw.pick(['ecog'])
        # drop bad channels
        bad_chan = raw.info['bads']
        raw = raw.drop_channels(raw.info['bads'])
        # epoch dataset
        epochs = self._epoch(raw, bad_chan)
        # delete raw to free up memory
        del raw 
        # the coordinate frame of the montage
        montage = epochs.get_montage()
        # add fiducials to montage
        montage.add_mni_fiducials('data')
        # set new montage
        epochs.set_montage(montage)
        # interpolate stimulation artifact
        epochs.apply_function(
            self._interpolate_stim_artifact, times=epochs[0].times,
            channel_wise=False
        )
        # apply baseline (z-score) normalization
        epochs.apply_function(
            rescale, times=epochs[0].times, 
            baseline=(self.baseline_range[0], self.baseline_range[1]), 
            mode='zscore', verbose=False
        )
        # get KDTree to query dist b/w stim and response electrodes
        ch_names = [c.strip().lower() for c in epochs.ch_names]
        el_kdtree = self._el_dist_nn(ch_names)
        # Nan filter stim electrodes and nearby electrodes per epoch
        id_to_event = {i: e for e,i in epochs.event_id.items()}
        stim_labels = [id_to_event[e] for e in epochs.events[:,2]]
        epochs.apply_function(
            self._nan_channel, ch_labels=ch_names, 
            stim_labels=stim_labels, el_kdtree=el_kdtree,
            channel_wise=False
        )
        # resample epoch object
        epochs = epochs.resample(self.resample)
        # create metadata
        epochs = self._create_metadata(epochs, stim_labels)

        return epochs




        












        
