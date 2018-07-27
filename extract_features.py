import numpy as np 
import mfanalysis as mf 
import utils
import config as cfg
import h5py
import os
from joblib import Parallel, delayed



current_dir = os.path.dirname(os.path.abspath(__file__))


N_JOBS = 4

def extract_features(args):
    """
    args = tuple containing: 
        file_idx : index of the file to be analyzed
        condition: 'interictal' or 'preictal'
        subject  : 'Dog_1', 'Dog_2' etc 
        n_channels: int config.info[subject]['n_channels']
        params   : dictionary config.mf_params[subject]
        p_info   : element of config.p_list, a tuple containing (p, p_index), the value of 
                   p and its index
    """

    file_idx, condition, subject, n_channels, params, p_info = args

    try:
        p = p_info[0]
        p_index = p_info[1]

        print("--- Analyzing file (%s, %s, %d)"%(subject, condition, file_idx))

        # Create MF object
        mfa   = mf.MFA(**params)
        mfa.p = p


        # Load data
        if condition == 'interictal':
            contents = utils.get_interictal_data(file_idx, subject)
        elif condition == 'preictal':
            contents = utils.get_preictal_data(file_idx, subject)

        data     = contents['data']
        sequence = contents['sequence']


        # Run MF analysis
        hurst = np.zeros(n_channels)
        c1 = np.zeros(n_channels)
        c2 = np.zeros(n_channels)
        c3 = np.zeros(n_channels)
        c4 = np.zeros(n_channels)

        for ii in range(n_channels):
            signal = data[ii, :]
            mfa.analyze(signal)

            cp  = mfa.cumulants.log_cumulants
            structure_dwt = mf.StructureFunction(mfa.wavelet_coeffs,
                                                 [2],
                                                 mfa.j1,
                                                 mfa.j2_eff,
                                                 mfa.wtype)
            
            hurst[ii]  = structure_dwt.zeta[0]/2
            c1[ii] = cp[0]
            c2[ii] = cp[1]
            c3[ii] = cp[2]
            c4[ii] = cp[3]


        # Save file
        output_dir = os.path.join(current_dir, 'cumulant_features', subject)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_filename = ('cumulants_%s_%d_p_%d.h5'%(condition, file_idx, p_index))
        output_filename = os.path.join(output_dir, output_filename)


        with h5py.File(output_filename, "w") as f:
            f.create_dataset('hurst', data = hurst )
            f.create_dataset('c1', data = c1 )
            f.create_dataset('c2', data = c2 )
            f.create_dataset('c3', data = c3 )
            f.create_dataset('c4', data = c4 )
            f.create_dataset('sequence', data = sequence )

        # return mfa
        # return hurst, c1, c2, c3, c4
    except:
        pass


if __name__ == '__main__':


    # hurst, c1, c2, c3, c4 = extract_features((1, 'preictal', 'Dog_2', cfg.info['Dog_2']['n_channels'], cfg.mf_params['Dog_2'], cfg.p_list[0]))


    arg_instances = []
    for subject in cfg.subjects:
        for condition in ['interictal', 'preictal']:
            if condition == 'interictal':
                indexes = cfg.info[subject]['interictal_files_idx']
            elif condition == 'preictal':
                indexes = cfg.info[subject]['preictal_files_idx']

            for file_idx in indexes:
                for p_info in cfg.p_list:
                    arg_instances.append((file_idx, 
                                          condition, 
                                          subject, 
                                          cfg.info[subject]['n_channels'], 
                                          cfg.mf_params[subject], 
                                          p_info
                                        ))


    # remove already computed instances
    new_arg_instances = []
    for args in arg_instances:
        file_idx, condition, subject, n_channels, params, p_info = args

        output_dir = os.path.join(current_dir, 'cumulant_features', subject)
        output_filename = ('cumulants_%s_%d_p_%d.h5'%(condition, file_idx, p_info[1]))
        output_filename = os.path.join(output_dir, output_filename)
        if os.path.isfile(output_filename):
            continue
        else:
            new_arg_instances.append(args)

    arg_instances = new_arg_instances


    Parallel(n_jobs=N_JOBS, verbose=1, backend="multiprocessing")(map(delayed(extract_features), arg_instances))
