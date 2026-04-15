#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <string.h>


enum Operator{
    non,
    mul,
    add,
    exp,
};

typedef struct  Value {
    float data;
    struct Value* left;
    struct Value* right;
    bool has_children;
    bool istrainable;
    bool visited;
    float grad;
    enum Operator operator;  
} Value;


typedef struct{
    int nrows;
    int ncols;
    Value** data;
} Matrix;

typedef struct {
    int n_neurons;
    int params_per_neuron;
    Matrix params;
    
} Layer;

typedef struct {
    Layer* layers;
    int n_layers;
    int n_comp;
} FF;

void ReLU(Matrix m){
    for (int i =0; i < m.nrows; i ++){
       for (int j=0;j < m.ncols; j++){
            if (m.data[i][j].data < 0){
                m.data[i][j].data = 0;
            }
        
       }
    }
   
 
}

Value*  mse(Value** preds , Value* y,int size , Value* loss){
  
    for (int i =0; i < size ;  i ++){
        Value* left = malloc(sizeof(Value));
        Value* right = malloc(sizeof(Value));
        
        Value* neg_y_l = malloc(sizeof(Value));
        *neg_y_l = y[i];
        neg_y_l->data = -y[i].data;
        neg_y_l->istrainable = false;
        Value* neg_y_r = malloc(sizeof(Value));
        *neg_y_r = y[i];
        neg_y_r->data = -y[i].data;
        neg_y_r->istrainable = false;

        left->left = &preds[i][0];
        left->right = neg_y_l;
        left->operator = add;
        left->has_children = true;
        left->istrainable = false;
        left->visited = false;
        left->grad = 0.0f;
      

        right->left = &preds[i][0];
        right->right = neg_y_r;
        right->operator = add;   
        right->has_children = true;
        right->istrainable = false;
        right->visited = false;
        right->grad = 0.0f;
        
        left->data = preds[i][0].data + neg_y_l->data;
        right->data = preds[i][0].data + neg_y_r->data;

       Value* mulres = malloc(sizeof(Value));
       mulres->has_children = true;
       mulres->istrainable = false;
       mulres->visited = false;
       mulres->grad = 0.0f;
       mulres->left = left;
       mulres->right = right;
       mulres->data = left->data * right->data;
       mulres->operator = mul;

       Value* new_loss = malloc(sizeof(Value));
       new_loss->istrainable = false;
       new_loss->has_children = true;
       new_loss->visited = false;
       new_loss->grad = 0.0f;
       new_loss->left = loss;
       new_loss->right = mulres;
       new_loss->operator = add;
       new_loss->data = new_loss->left->data + new_loss->right->data;

       loss = new_loss;
       

    }
    loss->data = loss->data / size;
    return loss;
  
}

int n_samples = 10;
int x_dim = 4;

int n_layers = 5;
int neurons_per_layer[] = {50,32,16,8,1};
 

float rand_float(float min, float max){
    return min + (float) rand() / RAND_MAX * (max - min);
}

void print_vector(Value* vector , int size ){
    printf("\n");
    for (int i =0; i < size; i ++){
        if (i ==0){
            printf("[");
        }
        printf("%f",vector[i].data);
        if (i == size - 1){
            printf("]");
       }else{
        printf(",");
       }
       
    }
    printf("\n");
}

void prepare_data(Value** X, Value* y ){
    for (int i =0; i < n_samples ; i ++){
        X[i] = malloc(x_dim * sizeof(Value));
        float even_ind_sums = 0;
        float odd_ind_sums = 0; 
        for (int j =0; j < x_dim; j++){
            Value w = {0}; 
            w.istrainable = false;
            w.has_children = false;
            w.visited = false;
            w.grad = 0.0f;
            w.operator = non;
            w.data = rand_float(-10,10);
            
            X[i][j] = w;

            if (j % 2 ==0){
                even_ind_sums += (X[i][j]).data;
            }else{
                odd_ind_sums += (X[i][j]).data;
            }
        }
        Value w = {0}; 
        w.istrainable = false;
        w.has_children = false;
        w.visited = false;
        w.grad = 0.0f;
        w.operator = non;
        w.data = even_ind_sums * odd_ind_sums  -  3.14*odd_ind_sums;

        y[i] = w;
        
    }
}


void transpose (Matrix mat , Matrix res){
    int nrows = mat.nrows;
    int ncols = mat.ncols;
    for (int i =0 ; i < nrows; i ++){
        for (int j=0; j < ncols ; j++){
            res.data[j][i] = mat.data[i][j];
        }
    }
}


// int n_comp = 0;
void matmul (Matrix m1 , Matrix m2, Matrix res){
    
    int m1_ncols = m1.ncols; 
    int m1_nrows = m1.nrows;
    int m2_ncols = m2.ncols;
    int m2_nrows = m2.nrows;
    // printf(" %d,%d\n",res.nrows,res.ncols);
    // printf(" %d,%d\n",m2_nrows,m2_ncols);
    
    if (m1_ncols != m2_nrows){
        printf("%s %d,%d\n","Invalid matrix multiplication",m1_ncols,m2_nrows);
        exit(0);
    }
  
    for (int i =0; i < m1_nrows;i ++){
        for (int j=0;j < m2_ncols; j++){
            Value v = {0};
            res.data[i][j]=v;
        }
    }

    for (int i =0; i < m1_nrows; i ++){
        for (int k = 0 ; k < m1_ncols;k++){
            for (int j=0 ; j < m2_ncols; j ++){
                Value left = res.data[i][j]; 
                
                Value *heap_mul = malloc(sizeof(Value));
                Value *heap_add_l = malloc(sizeof(Value));
                if (!heap_mul || !heap_add_l) {
                    free(heap_mul);
                    free(heap_add_l);
                    fprintf(stderr, "matmul: malloc failed\n");
                    exit(1);
                }

                *heap_add_l = left;

                heap_mul->data = m1.data[i][k].data * m2.data[k][j].data;
                heap_mul->left = &m1.data[i][k];
                heap_mul->right = &m2.data[k][j];
                heap_mul->has_children = true;
                heap_mul->operator = mul;
                heap_mul->grad = 0;
                heap_mul->istrainable = false;
                heap_mul->visited = false;

                Value v;
                v.data = left.data + heap_mul->data;
                v.left = heap_add_l;
                v.right = heap_mul;
                v.has_children = true;
                v.operator = add;
                v.grad = 0;
                v.istrainable = false;
                v.visited = false;

                res.data[i][j] = v;
                // n_comp += 4;
            }
        }
    }
}


void zero_grad(Value** topo, int sz){
    for (int i =0;i < sz; i ++){
        topo[i]->grad =0.0;
    }
    topo[sz-1]->grad  = 1;
}


int main(){
    
    Value** X = malloc(n_samples * sizeof(Value *));
    Value* y = malloc(n_samples * sizeof(Value));    
    Layer* layers = malloc(n_layers * sizeof(Layer));
    FF ff;
    ff.n_comp = 0;

    
   
    for (int i =0; i < n_layers; i ++){
        int params_per_neuron = 0;
        if (i ==0){
            params_per_neuron = x_dim;
        }else{
            params_per_neuron = layers[i-1].n_neurons;
        }
    
        Value** layer_params = malloc(neurons_per_layer[i]*sizeof(Value *));
        for (int j=0; j <neurons_per_layer[i]; j ++){
            Value* params = malloc(params_per_neuron * sizeof(Value)); 
            for (int k =0; k < params_per_neuron; k ++){
                Value p = {0};

                p.istrainable = true;
                p.data = rand_float(-1,1);
                params[k] = p;
            }
            layer_params[j] =  params;
            
        }

        Layer layer;

        Matrix layer_params_mat;
        layer_params_mat.nrows = neurons_per_layer[i];
        layer_params_mat.ncols = params_per_neuron;
        layer_params_mat.data = layer_params;

        layer.params = layer_params_mat;

        layer.n_neurons = neurons_per_layer[i];
        layer.params_per_neuron = params_per_neuron;
        layers[i] = layer;
    }


    ff.layers = layers;
    ff.n_layers = n_layers;

    prepare_data(X,y);
    
    //feed forward 
    Matrix outputs[n_layers + 1];
    Matrix X_mat;
    X_mat.data = X;
    X_mat.nrows = n_samples;
    X_mat.ncols = x_dim;
    outputs[0] = X_mat;



    // training 

int cap = 1024;
int initial_cap = 1024;
Value** topo = malloc(cap * sizeof(Value*));
    

Matrix transp_mats[n_layers];

for (int m=0; m < 10000; m ++){
    
    for (int i =1; i < n_layers + 1; i ++){

        Value** transp_res = malloc(ff.layers[i-1].params.ncols * sizeof(Value *));
        for (int q = 0; q < ff.layers[i-1].params.ncols; q++){
            transp_res[q] = malloc(ff.layers[i-1].params.nrows * sizeof(Value));
        }

        
        Matrix res_mat;
        res_mat.nrows = ff.layers[i-1].params.ncols;
        res_mat.ncols = ff.layers[i-1].params.nrows;
        res_mat.data = transp_res;

       
        transpose(ff.layers[i-1].params,res_mat);
        transp_mats[i-1] = res_mat;
        // printf("%d",i);

        Value** matmul_res = malloc(outputs[i-1].nrows * sizeof(Value *));
        for (int q = 0; q < outputs[i-1].nrows; q++){
            matmul_res[q] = malloc(res_mat.ncols * sizeof(Value));
        }

          
        Matrix matmul_res_mat;
        matmul_res_mat.nrows = outputs[i-1].nrows;
        matmul_res_mat.ncols = res_mat.ncols;
        matmul_res_mat.data = matmul_res;

        
        
        matmul(outputs[i-1],res_mat,matmul_res_mat);
        //apply relu on all hidden layers
        if (i < n_layers) ReLU(matmul_res_mat);
        outputs[i] = matmul_res_mat;

    }

    Matrix pred = outputs[n_layers];
    Value loss = {0};
    loss.data = 0.0f;
    loss.has_children = false;
    loss.istrainable = false;
    loss.visited = false;
    loss.grad = 0.0f;
    loss.operator = non;
    
    Value* loss_ptr = mse(pred.data,y,n_samples,&loss);
    printf("Loss : %f ",loss_ptr->data);
    
   
    
    
    
    
    printf("%i\n",ff.n_comp);
    //topological sort 
    void build_topo(Value* node, Value*** topo, int* sz){
        if (!node || node->visited) return;
        node->visited = true;
        if (node->has_children) {
            build_topo(node->left,topo,sz);
            build_topo(node->right,topo,sz);
        }

        if (*sz >= cap){
            cap *= 2;
            *topo = realloc(*topo,(cap) * sizeof(Value*));
            // if (!topo)    
        }
        (*topo)[(*sz)++] = node;
    }
   
    int sz = 0;
    build_topo(loss_ptr , &topo , &sz);
    zero_grad(topo,sz);
    void backprop(){
        for (int i =sz - 1; i >= 0 ; i --){
            
            Value* node = topo[i];
            topo[i]->visited = false;

            if (!node->has_children) continue;

            //a*b = n  
            if (node->operator == add){
                //dn/da = 1
                //dn / db = 1
                node->left->grad += 1 * node->grad;
                node->right->grad += 1 * node->grad;
            }else if (node->operator == mul){
                //dn/da = b 
                //dn/db = a 
                node->left->grad += (node->right->data)*node->grad;
                node->right->grad +=( node->left->data)*node->grad;
            } 
         }  
        }

    backprop();
    float lr = 1e-6;
    void step(){
        for (int i =0; i < sz ; i ++){
            if (topo[i]->istrainable){
                topo[i]->data -= lr*topo[i]->grad;
                // printf("grad %f",topo[i]->grad);
            }
    }
    }

    step();

    // sync updated weights back to original params
    for (int i = 0; i < n_layers; i++) {
        for (int j = 0; j < ff.layers[i].n_neurons; j++) {
            for (int k = 0; k < ff.layers[i].params_per_neuron; k++) {
                ff.layers[i].params.data[j][k].data = transp_mats[i].data[k][j].data;
            }
        }
    }







    
    for (int i = 1; i <= n_layers; i++) {
        for (int r = 0; r < outputs[i].nrows; r++) {
            for (int c = 0; c < outputs[i].ncols; c++) {
                Value* cell = &outputs[i].data[r][c];
                if (!cell->has_children) continue;
        
                Value* mul_node = cell->right;
                Value* add_node = cell->left;
                free(mul_node);
                while (add_node != NULL) {
                    if (!add_node->has_children) {
                      
                        free(add_node);
                        break;
                    }
                    Value* next_add = add_node->left;
                    Value* next_mul = add_node->right;
                    free(next_mul);
                    free(add_node);
                    add_node = next_add;
                }
            }
        }
    }


    {
        Value* cur = loss_ptr;
        while (cur != &loss) {
            Value* prev = cur->left;
            Value* mulres = cur->right;
            Value* diff_l = mulres->left;
            Value* diff_r = mulres->right;
            free(diff_l->right);   // neg_y_l
            free(diff_r->right);   // neg_y_r
            free(diff_l);
            free(diff_r);
            free(mulres);
            free(cur);
            cur = prev;
        }
    }

    // free transpose matrices
    for (int i = 0; i < n_layers; i++) {
        for (int q = 0; q < transp_mats[i].nrows; q++) {
            free(transp_mats[i].data[q]);
        }
        free(transp_mats[i].data);
    }


    for (int i = 1; i <= n_layers; i++) {
        for (int q = 0; q < outputs[i].nrows; q++) {
            free(outputs[i].data[q]);
        }
        free(outputs[i].data);
    }


    if (cap > initial_cap * 4) {
        cap = initial_cap;
        free(topo);
        topo = malloc(cap * sizeof(Value*));
    }

    
}
   


    // printf("%i",sz);

    // --- final cleanup ---
    free(topo);
    for (int i = 0; i < n_samples; i++) {
        free(X[i]);
    }
    free(X);
    free(y);
    for (int i = 0; i < n_layers; i++) {
        for (int j = 0; j < ff.layers[i].n_neurons; j++) {
            free(ff.layers[i].params.data[j]);
        }
        free(ff.layers[i].params.data);
    }
    free(layers);

    return 0;
}

