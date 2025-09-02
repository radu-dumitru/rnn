from data_loader import DataLoader

data_loader = DataLoader()
data = data_loader.load_data()

print(data["data_size"])