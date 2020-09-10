import numpy as np
import torch
import torch.nn as nn
import superp


# generate script for redlog input

def script_gen(barr_nn, ctrl_nn):
    def print_nn(model, model_str):
        layer = 0
        for p in model.parameters():
            layer = layer + 1
            arr = p.detach().numpy()
            if arr.ndim == 2:
                file.write(model_str + str(layer) + " := mat((")
                file.write('),\n('.join([   ', '.join(str(curr_int) for curr_int in curr_arr) for curr_arr in arr]))
                file.write("))$\n\n")
            elif arr.ndim == 1:
                file.write(model_str + str(layer) + " := tp mat((")
                file.write(', '.join(str(i) for i in arr))
                file.write("))$\n\n")
            else:
                print("Transform error!")

    with open("nnredlog.txt", "w") as file:
        ## out put dimensions
        file.write("sys_dim := " + str(superp.DIM_S) + "$\n")
        file.write("barr_dim := " + str(superp.D_H_B) + "$\n")
        file.write("ctrl_dim := " + str(superp.D_H_C) + "$\n\n")

        ## output barrier and control nn models
        print_nn(barr_nn, "weight_barr_")        #output barrier nn
        print_nn(ctrl_nn, "weight_ctrl_")        #output control nn

        ## output intermediary variables
        file.write("input_var := mat(")
        for i in range(superp.DIM_S-1):
            file.write("(x" + str(i+1) + "), ")
        file.write("(x" + str(superp.DIM_S) + "))$\n\n")

        file.write("barr_output_hidden := mat(")
        for i in range(superp.D_H_B-1):
            file.write("(bho" + str(i+1) + "), ")
        file.write("(bho" + str(superp.D_H_B) + "))$\n\n")

        file.write("ctrl_output_hidden := mat(")
        for i in range(superp.D_H_C-1):
            file.write("(cho" + str(i+1) + "), ")
        file.write("(cho" + str(superp.D_H_C) + "))$\n\n")

        file.write("barr_deri_hidden := mat(")
        for i in range(superp.D_H_B-1):
            file.write("(bdh" + str(i+1) + "), ")
        file.write("(bdh" + str(superp.D_H_B) + "))$\n\n")

        file.write("ctrl_bound := " + str(superp.CTRL_OUT_BOUND) + "$\n")
        if superp.CTRL_OUT_BOUND < 1e16:
            file.write("flag_bound_ctrl := true$\n")
        else:
            file.write("flag_bound_ctrl := false$\n")    
            
        file.write("END$\n")
