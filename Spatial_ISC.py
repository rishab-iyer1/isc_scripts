import argparse
import os
from glob import glob
from os.path import join

from ISC_Helper import compute_isc, get_rois, plot_spatial_isc


def main(_args):
    # -------------------------------
    # File paths
    # -------------------------------

    data_dir_func = '/Volumes/BCI/Ambivalent_Affect/fMRI_Study/ISC_Data_cut/NuisanceRegressed'
    func_fns = glob(join(data_dir_func, 'P?.nii.gz')) + glob(join(data_dir_func, 'N?.nii.gz')) + \
        glob(join(data_dir_func, 'VR?.nii.gz')) + glob(join(data_dir_func, 'P??.nii.gz')) + \
        glob(join(data_dir_func, 'N??.nii.gz')) + glob(join(data_dir_func, 'VR??.nii.gz'))

    roi_mask_path = '/Volumes/BCI/Ambivalent_Affect/rois'
    all_roi_fpaths = glob(os.path.join(roi_mask_path, '*.nii.gz'))

    roi_selected = ['auditory', 'ACC', 'vmPFC', 'insula', 'visualcortex', 'amygdala', 'wholebrain']
        # ['auditory', 'visualcortex', 'ACC', 'vmPFC', 'vPCUN', 'aINS_L', 'aANG_L', 'pANG_L', 'Insular_R',
        #             'dPCUN', 'aANG_R', 'aCUN', 'pANG_R', 'PMC_L', 'dPCC', 'insula', 'amygdala', 'wholebrain']

    # -------------------------------
    # Compute and display
    # -------------------------------

    if not os.path.exists('../data/iscs_roi_dict.pkl') or _args.redo:
        all_roi_masker = get_rois(all_roi_fpaths)
        print('spatial:', _args.spatial, 'pairwise:', _args.pairwise)
        compute_isc(roi_selected, all_roi_masker, func_fns, spatial=_args.spatial, pairwise=_args.pairwise)
        if _args.spatial:
            plot_spatial_isc(roi_selected)
        else:
            print("Please run with --spatial flag to plot spatial ISC")

    else:
        if _args.spatial:
            plot_spatial_isc(roi_selected)
        else:
            print("Please run with --spatial flag to plot spatial ISC")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute spatial ISC')
    parser.add_argument('--redo', action='store_true')
    parser.add_argument('--spatial', action='store_true', default=False)
    parser.add_argument('--pairwise', action='store_true', default=False)
    args = parser.parse_args()
    main(args)
