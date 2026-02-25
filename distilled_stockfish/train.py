import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch import optim, nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from pathlib import Path
from ChessPositionDataset import ChessPositionDataset
from LuessModel import LuessModel
from .WinProbLoss import WinProbLoss
import wandb



if __name__ == "__main__":

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using: {device}")

    
    BASE_DIR = Path(__file__).resolve().parent.parent


    BATCH_SIZE = 1024
    LEARNING_RATE = 2e-5
    EPOCHS = 30
    TRAINING_PERC = 0.8
    SEED = 0
    WEIGHT_DECAY = 1e-2
    NUM_RESIDUAL_BLOCKS = 12
    MAX_LEARNING_RATE = 5e-4
    MAX_NORM = 2.0
    NUM_WORKERS= 4



    run = wandb.init(
        entity="jakob-fanselow-hasso-plattner-institut",
        project="Luess",
        config={
            "learning_rate": LEARNING_RATE,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "weight_decay": WEIGHT_DECAY,
            "res_blocks:": NUM_RESIDUAL_BLOCKS,
            "max_lr": MAX_LEARNING_RATE,
            "max_norm": MAX_NORM
    },
)

    dataset = ChessPositionDataset(f"{BASE_DIR}/data/stockfishlabel_train.h5")

    generator = torch.Generator().manual_seed(SEED)
    train_dataset, test_dataset = random_split(dataset, [TRAINING_PERC, 1-TRAINING_PERC], generator=generator)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True,drop_last=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True,drop_last=True)

    model = LuessModel(num_res_blocks=NUM_RESIDUAL_BLOCKS).to(device)
    criterion = WinProbLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=MAX_LEARNING_RATE, 
                                                steps_per_epoch=len(train_dataloader), 
                                                epochs=EPOCHS)
    train_losses = []
    test_losses = []
    for epoch in range(EPOCHS):
        clipped = 0
        model.train()
        traing_progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

        train_running_loss = 0.0
        
        for data, labels in traing_progress_bar:
            data, labels = data.to(device), labels.to(device)


            optimizer.zero_grad()          
            outputs = model(data)          
            loss = criterion(outputs, labels)
            loss.backward()       
            old_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_NORM)      
            if old_norm > MAX_NORM:
                clipped += 1    
            optimizer.step()

            scheduler.step()
            train_running_loss += loss.item()

        print(f"Clipped percentage: {clipped/len(train_dataloader)}")

        train_loss = train_running_loss / len(train_dataloader)
        print(f"train_loss: {train_loss}")

        train_losses.append(train_loss)
        

        model.eval()
        test_running_loss = 0

        with torch.no_grad():
            for  data, labels in test_dataloader:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data) 
                loss = criterion(outputs, labels)

                test_running_loss += loss.item()
        test_loss = test_running_loss / len(test_dataloader)
        print(f"test_loss: {test_loss}")

        test_losses.append(test_loss)

        run.log({"train_loss": train_loss, "test_loss": test_loss, "clipped_percentage": clipped/len(train_dataloader)})
    

    run.finish()
    torch.save(model.state_dict(), 'model_weights.pth')
    epoch_range = range(1, len(test_losses) + 1)



    plt.plot(epoch_range, train_losses, label='Training Loss', color='blue', linestyle='-')
    plt.plot(epoch_range, test_losses, label='Test Loss', color='orange', linestyle='--')

    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.savefig('loss.png')

