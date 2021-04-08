from dataloader import L2RDataset

if __name__ == '__main__':
    train_file = 'Toda_201911.txt'
    train_ds = L2RDataset(file=train_file, data_id='BOATRACE')

    for batch_rankings, batch_std_labels in train_ds:
        print(batch_rankings)
        print(batch_std_labels)

