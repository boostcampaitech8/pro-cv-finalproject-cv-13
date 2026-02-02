# Tumor Segmentation Inference (nnUNetv2)

Minimal single-case inference pipeline:
- preprocess (CT/PT -> resample + ROI)
- nnUNetv2 prediction (model folder)
- restore to raw resolution
- optional Dice score if label is present

## Prereqs
- Python environment with nnUNetv2 and deps
- install requirements.txt
- `cd STU-Net` and `pip install -e .`
- `nnUNetv2_predict_from_modelfolder` in PATH (or pass `--nnunet-predict`)

## Basic usage
```
python tumor_segmentation/inference.py ../test_dir/CHUM-001/CHUM-001__CT.nii.gz outputs/CHUM-001 --model-folder tumor_segmentation/plans --checkpoint checkpoint_best.pth --pt-path ../test_dir/CHUM-001/CHUM-001__PT.nii.gz
```

## Notes
- `--pt-path` is optional. If omitted, the script looks for `*__PT.nii.gz` in the same folder as CT.
- If a single label file exists in the input folder (not CT/PT), Dice will be printed.
- Output is written to `outputs/CHUM-001/CHUM-001.nii.gz` in the example above.

## Optional args
- `--label-path PATH` : explicit GT label path for Dice.
- `--nnunet-predict PATH` : path to `nnUNetv2_predict_from_modelfolder`.
- `--folds "0 1 2 3 4"` : folds to use.