import soundfile as sf

path = "C:/Users/bi1ahj/Desktop/Datasets/Data_for_hierarchical_mapping/Audio/230322_Nest_14_R6.WAV"

print(sf.info(path))

with open(path, "rb") as f:
    print(f.read(20))