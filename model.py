import torch
import torch.nn as nn

class SDN(nn.Module):
    def __init__(self, input, layer_1, layer_2, output, group, alpha) -> None:
        super().__init__()
        self.layer_1 = layer_1
        self.layer_2 = layer_2
        self.output = output
        self.group = group
        self.alpha = alpha
        self.hidden1 = layer_1 * group
        self.hidden2 = layer_2 * group
        
        self.fc1 = nn.Linear(input, self.hidden1)
        self.fc2 = nn.Linear(self.hidden1, self.hidden2)
        self.fc3 = nn.Linear(self.hidden2, output)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.SDA(x)
        x = self.fc2(x)
        x = self.SDA(x)
        x = self.fc3(x)
        return x
        
    def SDA(self, x):
        for idx in range(len(x)):
            active, inactive = -1, -1 # Record the index of active/inactive door
            group_number = len(x[idx]) // self.group # Total number of groups
            
            # Check each group from left to right
            for group_index in range(group_number):
                
                # Active door check
                checker = True if active <0 else False
                index_to_check = range(group_index * self.group, (group_index+1) * self.group)
                for i in index_to_check:
                    if x[idx, i] <= 0:
                        checker = False
                        break
                if checker:
                    # In this case, all x[i]>0, activate door is found
                    active = group_index
                                            
                
                # Inactive door check
                checker = True if inactive <0 else False
                index_to_check = range(group_index * self.group, (group_index+1) * self.group)
                for i in index_to_check:
                    if x[idx, i] >= 0:
                        checker = False
                        break
                if checker:
                    # In this case, all x[i]<0, inactivate door is found                        
                    inactive = group_index
                
                if active>=0 and inactive>=0:
                    break
                
            if active != -1:
                # valid activate door found
                index_to_activate = range(active * self.group, (active+1) * self.group)
                x[idx, index_to_activate] *= self.alpha
            if inactive != -1:
                # valid inactivate door found
                index_to_inactivate = range(inactive * self.group, (inactive+1) * self.group)
                x[idx, index_to_inactivate] *= 0
                
            return x