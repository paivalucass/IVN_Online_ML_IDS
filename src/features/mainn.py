
import numpy as np

y_data = np.array([[0,"Normal"],[0,"Normal"], [0,"CD"], [0,"Normal"], [0,"Normal"], [0,"CD"], [0,"CD"], [0,"CD"], [0,"CD"], [0,"Normal"], [0,"Normal"]])
x_data = np.array(["0","0","0","0","0","0","0","0","0","0","0","0"])

print(y_data.shape)

attack = "CD"

index_initial = 0
index_final = 0
counter = 0 

labels = y_data[:,1]

print(labels)

attack_count = np.sum(labels == attack)

print(attack_count)

for i, label in enumerate(labels):
    if label == attack:
        counter += 1
        if counter == 1:
            index_initial = i
        elif counter == attack_count:
            index_final = i+1
            break
        
        
print(index_initial)
print(index_final)
        
print(f"!!! DELETED {attack_count} {attack} FROM DATASET !!!")
            
x_data = np.delete(x_data, np.s_[index_initial: index_final], axis=0)
y_data = np.delete(y_data, np.s_[index_initial: index_final], axis=0)    

print(x_data)
print(y_data)  