import torch

configurations = {
    1: dict(
        # Reproducibility
        SEED=1337,

        # CASIA-WebFace
        DATA_ROOT="/content/CASIA-WebFace/data",
        RECORD_DIR="/content/CASIA-WebFace/data/casia_train.txt",

        # Persistent Google Drive storage
        MODEL_ROOT="/content/drive/MyDrive/ABSS-CurricularFace-Implementation/checkpoints",
        LOG_ROOT="/content/drive/MyDrive/ABSS-CurricularFace-Implementation/logs",

        # Resume configuration
        RESUME_PATH="",

        # Architecture
        BACKBONE_NAME="IR_101",
        HEAD_NAME="CurricularFace",
        LOSS_NAME="Softmax",
        INPUT_SIZE=[112, 112],
        RGB_MEAN=[0.5, 0.5, 0.5],
        RGB_STD=[0.5, 0.5, 0.5],
        EMBEDDING_SIZE=512,

        # Training
        BATCH_SIZE=64,
        LR=0.1,
        START_EPOCH=0,
        NUM_EPOCH=50,
        WEIGHT_DECAY=5e-4,
        MOMENTUM=0.9,
        STAGES=[28, 38, 46],

        # Single GPU
        GPU=0,
        NUM_WORKERS=2,

        # Checkpointing
        SAVE_EVERY_EPOCH=1,
        SAVE_EVERY_STEPS=500,
    ),
}
