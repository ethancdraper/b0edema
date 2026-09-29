from file_tree import FileTree
from fsl_pipe import Pipeline, In, Out, Ref

# Load some libraries that allow us to run FSL tools
from fsl import wrappers
from subprocess import run
from os import getenv

from fsl_pipe import Pipeline, In, Out, Ref, Var
from file_tree import FileTree
import fsl
from fsl.data.image import Image
from fsl.wrappers import flirt, bet, fast, fslmaths, applyxfm, fslstats
from fsl.wrappers import wrapperutils as wutils
from fsl.utils import assertions as asrt
import numpy as np
import os

# Trying to write a wrapper to call fslmeants
@wutils.fileOrImage('input', 'output')
@wutils.fslwrapper
def fslmeants(input, output, *args):
    """Wrapper for the ``fslmeants`` tool."""

    asrt.assertIsNifti(input)

    cmd = ['fslmeants', input, output] + [str(a) for a in args]

    return cmd

# Trying to write a wrapper to call select_dwi_vols
@wutils.fileOrImage('input', 'output')
@wutils.fslwrapper
def select_dwi_vols(input, bvals, output, approx_bval, *args):
    """Wrapper for the ``select_dwi_vols`` tool."""

    asrt.assertIsNifti(input)

    cmd = ['select_dwi_vols', input, bvals, output, str(approx_bval)] + [str(a) for a in args]

    return cmd

# Load the file-tree describing the data directory
tree = FileTree.read("data.tree").update_glob("T1w", link=[("subject", "ses")])

# Create the pipeline
pipe = Pipeline()

# Add recipes to the pipeline
# Filled by user
@pipe
def preproc1(T1w: In, T1w_brain: Out, T1w_brain_mask: Out):
    """
    Stage 1 of oedema_pipe, preprocessing the input images.

    """
    print(f"Processing T1w")

    bet(T1w, T1w_brain, mask=T1w_brain_mask)

if __name__ == "__main__":
    pipe.cli(tree)

@pipe
def avb0(DWI: In, bvals: In, rB0: Out, B0: Out):
    """
    Averaging B0.
    """
    select_dwi_vols(DWI, bvals, rB0, approx_bval=0)
    fslmaths(rB0).Tmean().run(B0)

@pipe
def preproc2(T1w_brain: In, FLAIR: In, B0: In, FLAIR_to_T1_mat: Out, T1_to_B0_mat: Out):
    """
    Stage 2 of oedema_pipe, generating transformation matrices.

    """

    print(f"Performing registrations")

    if os.path.exists(T1w_brain):
        flirt(FLAIR, T1w_brain, omat=FLAIR_to_T1_mat)
        flirt(T1w_brain, B0, omat=T1_to_B0_mat)

    else:
            raise FileNotFoundError(f"Brain extracted T1w not found.")

@pipe
def preproc3(FLAIR_to_T1_mat: In, TUM: In, T1w_brain: In, T1w_brain_mask: In, TUM_in_T1: Out, TUM_bin: Out, inv_TUM: Out, T1w_NT: Out):
    """
    Stage 3 of oedema_pipe, applying registrations.

    """
    if os.path.exists(FLAIR_to_T1_mat):
        applyxfm(TUM, T1w_brain, FLAIR_to_T1_mat, out=TUM_in_T1)
        fslmaths(TUM_in_T1).thr(0.8).bin().run(TUM_bin)
        fslmaths(T1w_brain_mask).sub(TUM_bin).thr(0.8).bin().run(inv_TUM)
        fslmaths(T1w_brain).mas(inv_TUM).run(T1w_NT)
        print(f"Registration successful and applied to TUM, producing T1w_NT.")

    else:
        raise FileNotFoundError(f"Transformation matrix {FLAIR_to_T1_mat} not found.")

@pipe
def seg(T1w_NT: In, FAST: Ref):
    """
    Stage 4 of oedema_pipe, calculating oedema in b0.

    """
    
    print(f"Performing FAST segmentation on T1w_NT")

    #fast(T1w_NT, FAST).b.run()
    wrappers.fast(T1w_NT, FAST, b=True).run()
#def oedema_pipeline(T1w_NT: In, BF: In, B0: In, T1_to_B0_mat: In, WM: In, BF_in_B0: Out, B0_bias_corrected: Out, WM_in_B0: Out, WM_thr: Out, BF_in_B0_WM_sig: Out, OEDEMA: Out):
    """
    Stage 4 of oedema_pipe, calculating oedema in b0.

    """
    
    #print(f"Performing FAST segmentation on T1w_NT")

    #fast(T1w_NT, out=FAST, b=True)

#    if os.path.exists(BF):
#        print(f"FAST completed. Calculating oedema in b0")
#        applyxfm(BF, B0, T1_to_B0_mat, out=BF_in_B0)
#        fslmaths(B0).div(BF_in_B0).run(B0_bias_corrected)
#        applyxfm(WM, B0, T1_to_B0_mat, out=WM_in_B0)
#        fslmaths(WM_in_B0).thr(0.8).bin().run(WM_thr)
#        WM_sig = fslstats(B0_bias_corrected).k(WM_thr).M.run()
#        print(f"{sub}: WM_sig:", WM_sig)
#        fslmaths(BF_in_B0).mul(WM_sig).run(BF_in_B0_WM_sig)
#        fslmaths(B0).div(BF_in_B0_WM_sig).run(OEDEMA)
#    else:
#        raise FileNotFoundError(f"Bias field {BF} not found.")

if __name__ == "__main__":
    pipe.cli(tree)