from tkinter import *
from PIL import Image, ImageDraw, ImageGrab
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

class UserInterface:
    def __init__(self, width, height):
        self.window = Tk()
        self.window.title('Classifier')
        self.window.resizable(0, 0)
        self.canvas = Canvas(width=width, height=height, bg='black')
        self.canvas.bind('<B1-Motion>', self.draw_line)
        classify = Button(text='Run classifier on canvas',
                          command=lambda: Net.classify(Net())).pack(side=TOP)
        clear_canvas = Button(text='Clear Canvas',
                              command=lambda: self.canvas.delete('all')).pack(side=TOP)
        retrain_network = Button(
            text='Retrain network', command=lambda: Net().train_and_save_model()).pack(side=TOP)
        
    
    def draw_line(self, event):
        x, y = event.x, event.y
        self.canvas.create_oval((x-2, y-2, x+2, y+2), outline='white')

    def save_as_png(self, file_name):
        x = self.window.winfo_rootx() + self.canvas.winfo_x()
        y = self.window.winfo_rooty() + self.canvas.winfo_y()
        x1 = x + self.canvas.winfo_width()
        y1 = y + self.canvas.winfo_height()
        ImageGrab.grab().crop((x, y, x1, y1)).save(file_name)

    def draw(self):
        self.canvas.pack()
        self.window.mainloop()

ui = UserInterface(224, 224)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

    def train_(self, model, train_loader, optimizer, epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0:  # how many batches to wait before logging training status
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

    def test(self, model, test_loader):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                # sum up batch loss
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                # get the index of the max log-probability
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

    def train_and_save_model(self):
        train_kwargs = {'batch_size': 64}
        test_kwargs = {'batch_size': 1000}
        if torch.cuda.is_available():
            cuda_kwargs = {'num_workers': 1,
                           'pin_memory': True,
                           'shuffle': True}
            train_kwargs.update(cuda_kwargs)
            test_kwargs.update(cuda_kwargs)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        dataset1 = datasets.MNIST('../data', train=True, download=True,
                                  transform=transform)
        dataset2 = datasets.MNIST('../data', train=False,
                                transform=transform)
        train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
        test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
        
        model = Net().to(self.device)
        optimizer = optim.Adadelta(model.parameters(), lr=1.0)

        scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
        for epoch in range(1, 14 + 1):
            self.train_(model, train_loader, optimizer, epoch)
            self.test(model, test_loader)
            scheduler.step()

        torch.save(model.state_dict(), "model.pt")

    def classify(self):
        UserInterface.save_as_png(ui, "image.png")
        model = Net().to(self.device)
        model.load_state_dict(torch.load("model.pt"))
        transform = transforms.Compose([
        transforms.ToTensor(),  # convert the image to a tensor
        transforms.Normalize((0.1307,), (0.3081,))  # normalize the image
        ])
        
        image = Image.open("image.png").convert("L")  # open the image
        image = image.resize((28, 28))
        image = transform(image)  # apply the transformation
        image = image.unsqueeze(0)  # add a batch dimension
        
        output = model(image.to(self.device))
        _, pred = torch.max(output, dim=1)
        
        print(f"Prediction: {pred.item()}")
        ui.canvas.create_text(40, 20, text=f"pred: {pred.item()}", fill="white", font=('Helvetica 15 bold'))
        ui.canvas.pack()

def main():
    ui.draw()
    
if __name__ == '__main__':
    main()
