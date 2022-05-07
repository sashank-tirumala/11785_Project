README
Instruction of running this code: First, you need to download the dataset we create from the box. Change the root path in import section to the path of your dataset. Then you just need to run the code section by section.

Don't forget to modified the path of saving the model after training. Since this model only need half hour to converge, we don't save the checkpoint for each traing epoch. But you can modified that part.

Best Model Architecture
We have make more than 30 models for this problem. The best result is

    Network(
      (embedding): Sequential(
        (0): Conv1d(15, 128, kernel_size=(1,), stride=(1,), padding=(1,))
        (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
        (3): Dropout(p=0.3, inplace=False)
        (4): Conv1d(128, 256, kernel_size=(3,), stride=(1,), padding=(1,))
        (5): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (6): ReLU()
        (7): Dropout(p=0.3, inplace=False)
      )
      (lstm): LSTM(256, 512, batch_first=True, dropout=0.1, bidirectional=True)
      (classification): Sequential(
        (0): Linear(in_features=1024, out_features=2048, bias=True)
        (1): ReLU()
        (2): Dropout(p=0.3, inplace=False)
        (3): Linear(in_features=2048, out_features=5, bias=True)
      )
    )


For more detail about the architecture and the abliation study please read our paper.