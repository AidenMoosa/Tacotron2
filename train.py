from model import Tacotron2

tacotron2 = Tacotron2()

tacotron2.train()

for _ in range(10):
    print("henlo")

'''
for i in range(avd.__len__()):
    name, label = avd.__getitem__(i)

    input_tensor = line_to_tensor(label).unsqueeze(0)

    output, _ = encoder(input_tensor)
    print(output[0])
'''