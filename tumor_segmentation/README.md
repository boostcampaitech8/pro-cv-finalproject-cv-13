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
python tumor_segmentation/inference.py ../test_dir/CHUM-001/CHUM-001__0000.nii.gz outputs/CHUM-001 --model-folder tumor_segmentation/plans --checkpoint checkpoint_best.pth --pt-path ../test_dir/CHUM-001/CHUM-001__0001.nii.gz
```

## Docker
- When using Docker, comment out the torch installation in `requirements.txt`.
  
## Basic Docker Usage
```
docker build -t next-ct ./tumor_segmentation 
docker run --gpus all --shm-size=8g -v %cd%\..\test_dir:/test_dir -v %cd%\.\outputs:/app/outputs next-ct
```

## Notes
- nnUNet naming is expected: CT=`*_0000.nii.gz`, PT=`*_0001.nii.gz`, label=`*.nii.gz` (same case id).
- `--pt-path` is optional. If omitted, the script looks for `<CASE_ID>_0001.nii.gz` next to CT.
- If `<CASE_ID>.nii.gz` exists in the input folder, Dice will be printed automatically.
- Output is written to `outputs/CHUM-001/CHUM-001.nii.gz` in the example above.

## Optional args
- `--label-path PATH` : explicit GT label path for Dice.
- `--nnunet-predict PATH` : path to `nnUNetv2_predict_from_modelfolder`.
- `--folds "0 1 2 3 4"` : folds to use.
