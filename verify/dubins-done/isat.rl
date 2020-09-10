%% generate isat input for verification

load_package redlog$
rlset ofsf$
off nat$
off exp$
off echo$


%%%%%%%%%%%%%%%%%neural network specification%%%%%%%%%%%%%%%%%
in "nnredlog.txt"$

%% compute the barr nn output (without activation)
barr_input_hidden := weight_barr_1 * input_var + weight_barr_2$
barr_output := weight_barr_3 * barr_output_hidden + weight_barr_4$
barr_output := barr_output(1, 1)$

%% compute the ctrl nn output with ReLU activation function
ctrl_input_hidden := weight_ctrl_1 * input_var + weight_ctrl_2$
ctrl_output_hidden := ctrl_input_hidden$
for i:=1 step 1 until ctrl_dim do << ctrl_output_hidden(i, 1) := max(0, ctrl_input_hidden(i, 1)); >>$
ctrl_output := weight_ctrl_3 * ctrl_output_hidden + weight_ctrl_4$
%% the output layer of controller nn: Hardtanh
ctrl_output := ctrl_output(1, 1)$
if flag_bound_ctrl = true then <<ctrl_output := max(-ctrl_bound, min(ctrl_bound, ctrl_output))$>>$

%% compute the gradient of barrier nn wrt. input_var
matrix m_diag(barr_dim, barr_dim)$
for i:=1 step 1 until barr_dim do << m_diag(i, i) := barr_deri_hidden(i, 1); >>$
barr_dh_var := m_diag * weight_barr_1$
barr_dnn_var := weight_barr_3 * barr_dh_var$
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




%%%%%%%%%%%%%%%%%system specification%%%%%%%%%%%%%%%%%%%%%%%%
vector_field := mat((sin(x2), -ctrl_output))$
init := x1 >= -1 and x1 <= 1 and x2 >= -3.1416 / 16 and x2 <= 3.1416 / 16$
domain := x1 >= -6 and x1 <= 6 and x2 >= -0.7 * 3.1416 and x2 <= 0.7 * 3.1416$
unsafe := domain and (not (x1 >= -5 and x1 <= 5 and x2 >= -3.14159 / 2 and x2 <= 3.14159 / 2))$
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
lie := barr_dnn_var * (tp vector_field)$
lie := lie(1, 1)$
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%activation function specification%%%%%%%%%%%%%%%%%%%%%%%%
%% y = 0.5 * x + torch.sqrt(0.25 * x * x + 0.0001)

bent_6 := (f_in <= -0.1 and f_out > 0 and f_out < 0.0009902)
        or (f_in > -0.1 and f_in <= -0.05 and f_out > 0.0009901 and f_out < 0.001926)
        or (f_in > -0.05 and f_in <= 0 and f_out > 0.00414214 + 0.146447 * (0.02 + f_in) - 0.00000001 and f_out < 0.00414214 + 0.146447 * (0.02 + f_in) +  0.00293 and f_out > 0) 
        or (f_in > 0 and f_in < 0.05 and f_out > 0.0224536 + 0.834482 * (-0.018 + f_in) and f_out < 0.0224536 + 0.834482 * (-0.018 + f_in) + 0.00277) 
        or (f_in >= 0.05 and f_in < 0.1 and f_out > f_in + 0.0009901 and f_out < f_in + 0.001926)
        or (f_in >= 0.1 and f_out > f_in and f_out < f_in + 0.0009902)$

bent_4 := (f_in <= -0.05 and f_out > 0 and f_out < 0.001926)
        or (f_in > -0.05 and f_in <= 0 and f_out > 0.001925 and f_out <= 0.01)
        or (f_in > 0 and f_in < 0.05 and f_out > f_in + 0.001925 and f_out < f_in + 0.01)
        or (f_in >= 0.05 and f_out > f_in and f_out < f_in + 0.001926)$

bent_2 := (f_in <= 0 and f_out > 0 and f_out <= 0.01)
        or (f_in > 0 and f_out > f_in and f_out < f_in + 0.01)$

deri_bent :=  (f_in <= -0.1 and f_out > 0 and f_out < 0.00971) 
        or (f_in > -0.1 and f_in <= -0.05 and f_out > 0.009709 and f_out < 0.035762)
        or (f_in > -0.05 and f_in <= -0.02 and f_out > 0.0724011 + 3.48087 * (0.033 + f_in) and f_out < 0.0724011 + 3.48087 * (0.033 + f_in) + 0.0288 )
        or (f_in > -0.02 and f_in <= 0 and f_out > 0.242752 + 15.7627 * (0.012 + f_in) and f_out < 0.242752 + 15.7627 * (0.012 + f_in) + 0.0681 )
        or (f_in > 0 and f_in < 0.02 and f_out > 0.757248 + 15.7627 * (-0.012 + f_in) - 0.0681 and f_out < 0.757248 + 15.7627 * (-0.012 + f_in) )
        or (f_in >= 0.02 and f_in < 0.05 and f_out > 0.927599 + 3.48087 * (-0.033 + f_in) - 0.0288 and f_out < 0.927599 + 3.48087 * (-0.033 + f_in) )
        or (f_in >= 0.05 and f_in < 0.1 and f_out > 0.9642 and f_out < 0.9903)
        or (f_in >= 0.1 and f_out < 1 and f_out > 0.99028)$
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




%%%%%%%%%%%%%%%%%%%%%%application of bent relu activation%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
act_f := bent_6$

app_bent := true$
for i:=1 step 1 until barr_dim do << app_bent := app_bent and sub({f_out=barr_output_hidden(i, 1), f_in=barr_input_hidden(i, 1)}, act_f); >>$

app_deri_bent := true$
for i:=1 step 1 until barr_dim do << app_deri_bent := app_deri_bent and sub({f_out=barr_deri_hidden(i, 1), f_in=barr_input_hidden(i, 1)}, deri_bent); >>$
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%specification of verification conditions%%%%%%%%%%%%%%%%%%%%%%%
cons_init := init and app_bent and barr_output > 0$
cons_safe := unsafe and app_bent and barr_output <= 0$
cons_lie := domain and app_bent and barr_output = 0 and app_deri_bent and lie >= 0$
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%generate initial condition%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
out "init_veri.hy"$
write "DECL"$
write "real [-6, 6] x1"$
write "real [-5 / 2, 5 / 2] x2"$
for i:=1 step 1 until barr_dim do 
<< write "real [0, 10000000000] ", barr_output_hidden(i, 1); >>$
write "EXPR"$
write rlsimpl(cons_init)$
shut "init_veri.hy"$

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%generate unsafe condition%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
out "safe_veri.hy"$
write "DECL"$
write "real [-6, 6] x1"$
write "real [-2 - 1 / 4, 2 + 1 / 4] x2"$
for i:=1 step 1 until barr_dim do 
<< write "real [0, 10000000000] ", barr_output_hidden(i, 1); >>$
write "EXPR"$
write rlsimpl(cons_safe)$
shut "safe_veri.hy"$

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%generate lie condition%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
out "lie_veri.hy"$
write "DECL"$
write "real [-6, 6] x1"$
write "real [-5 / 2, 5 / 2] x2"$
for i:=1 step 1 until barr_dim do 
<< write "real [0, 10000000000] ", barr_output_hidden(i, 1); >>$
for i:=1 step 1 until barr_dim do 
<< write "real [0, 1] ", barr_deri_hidden(i, 1); >>$
write "EXPR"$
write rlsimpl(cons_lie)$
shut "lie_veri.hy"$