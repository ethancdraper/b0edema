# Oedema Pipeline

# Setup

from fsl_pipe import Pipeline, In, Out, Ref, Var
from file_tree import FileTree
import fsl
from fsl.data.image import Image
from fsl.wrappers import flirt, bet, fast, fslmaths, applyxfm, fslstats
import numpy as np
import os

# Pipeline
pipe = Pipeline()

@pipe

def preproc1(T1w: In, T1w_brain: Out, T1w_brain_mask: Out):
    """
    Stage 1 of oedema_pipe, preprocessing the input images.

    """
    print(f"Processing T1w")

    bet(T1w, T1w_brain, mask=T1w_brain_mask)

def preproc2(T1w_brain: In, FLAIR: In, B0: In, FLAIR_warp_basename: Ref, T1w_warp_basename: Ref, FLAIR_to_T1_mat: Out, T1_to_B0_mat: Out):
    """
    Stage 2 of oedema_pipe, generating transformation matrices.

    """

    print(f"Performing registrations")

    if os.path.exists(T1w_brain):
        flirt(FLAIR, T1w_brain, omat=FLAIR_to_T1_mat, logout=FLAIR_warp_basename + ".log")
        flirt(T1w_brain, B0, omat=T1_to_B0_mat, logout=T1w_warp_basename + ".log")

    else:
            raise FileNotFoundError(f"Brain extracted T1w not found.")

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

def oedema_pipeline(T1w_NT: In, BF: In, B0: In, T1_to_B0_mat: In, WM: In, BF_in_B0: Out, B0_bias_corrected: Out, WM_in_B0: Out, WM_thr: Out, BF_in_B0_WM_sig: Out, OEDEMA: Out):
    """
    Stage 4 of oedema_pipe, calculating oedema in b0.

    """
    
    print(f"Performing FAST segmentation on T1w_NT")

    fast(T1w_NT, out=FAST, b=True)

    if os.path.exists(BF):
        print(f"FAST completed. Calculating oedema in b0")
        applyxfm(BF, B0, T1_to_B0_mat, out=BF_in_B0)
        fslmaths(B0).div(BF_in_B0).run(B0_bias_corrected)
        applyxfm(WM, B0, T1_to_B0_mat, out=WM_in_B0)
        fslmaths(WM_in_B0).thr(0.8).bin().run(WM_thr)
        WM_sig = fslstats(B0_bias_corrected).k(WM_thr).M.run()
        print(f"{sub}: WM_sig:", WM_sig)
        fslmaths(BF_in_B0).mul(WM_sig).run(BF_in_B0_WM_sig)
        fslmaths(B0).div(BF_in_B0_WM_sig).run(OEDEMA)
    else:
        raise FileNotFoundError(f"Bias field {BF} not found.")

def save_sig(WM_Sig: float, WM_file: Out):
    """
        Save the WM_sig value to a text file.

    """
    with open(WM_file, "w") as f:
        f.write(f"WM_sig: {WM_Sig}\n")