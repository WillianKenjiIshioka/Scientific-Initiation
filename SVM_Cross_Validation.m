%SVM using cross validation strategy
clear all; close all; clc;

load('datasets'); %Data set containg features extratcted

datasetBase = datasetAceleracao;
%datasetBase = datasetTensao;
%datasetBase = datasetCorrenteTensao;

%Tratamento da matriz
for(j=1:length(datasetBase))
    if(datasetBase(j,11)~=3)
        datasetBase(j,11)=0;
    end
    if(datasetBase(j,11)==3)
        datasetBase(j,11)=1;
    end
    
end

output(:,1) = datasetBase(:,11); %Saída Dataset com 10 atributos e 2 classe

datasetBase = datasetBase(:,1:10); % Define a tabela de input a ser normalizada

dataset = zscore(datasetBase); % Normalizacao do input

input(:,1:10) = dataset(:,1:10); %Entrada Dataset com 10 atributos e 2 classe

%Faz partição para otimização de hyperparametros
cvp = cvpartition(output(:,1),'kfold',5); 


%Treinamento - SVM
svmStruct = fitcsvm(input,output,'Solver','SMO','KernelFunction',...
'rbf','OptimizeHyperparameters','auto',...
'HyperparameterOptimizationOptions',...
struct('AcquisitionFunctionName','expected-improvement-plus',...
'CVPartition',cvp,'ShowPlots',false));


%Roda SVM com Hyperparametros otimizados fazendo cross validacao 5 k-fold
for j=1 : cvp.NumTestSets %Quantidade de conjuntos de teste e treinamento
    trIdx = cvp.training(j);
    teIdx = cvp.test(j);
    
    Xtrain = input(trIdx,:);
    Ytrain = output(trIdx,1);
    
    Xtest = input(teIdx,:);
    Ytest = output(teIdx,1);
    
    %Ajuste de gama para sigma para utilizar na funcao kernel
    sigma = svmStruct.ModelParameters.KernelScale;
    gama = 1/(2*sigma^2);

    %Fator de suavizacao
    C = svmStruct.ModelParameters.BoxConstraint;

    %Treino da SVM

    %Treina SVM com hyperparametros otimizados e conjunto de treinamento
    svmStruct = fitcsvm(Xtrain,Ytrain,... 
    'Solver','SMO','KernelFunction','rbf','KernelScale',sigma,...
    'BoxConstraint',C);

    %Teste da SVM para conjutno de testes
    tic;   
        testClass = predict(svmStruct,Xtest);
    time = toc;

    %Grava os resultados obtidos
    err = 0;
    fp = 0; %False positive
    fn = 0; %False Negative
    tp = 0; %True positive
    tn = 0; %True Negative

    for k=1 : length(testClass)
         if(Ytest(k) ~= testClass(k))
             err = err + 1;
         end
         if (Ytest(k) == 0 && testClass(k) == 1)
             fp = fp + 1;
         end
         if (Ytest(k) == 1 && testClass(k) == 0)
             fn = fn + 1;
         end
         if (Ytest(k) == 1 && testClass(k) == 1)
             tp = tp + 1;
         end
         if (Ytest(k) == 0 && testClass(k) == 0)
             tn = tn + 1;
         end
    end
    
    taxa_acerto = 1-(err/length(testClass));
    
    result(j,1) = j; %Numero do conjunto de treinamento/teste
    result(j,2) = gama; %Valor se sigma da funcao kernel RBF
    result(j,3) = C; %Constante de Suavizacao
    result(j,4) = err; %N?mero de amostras classificadas incorretamente
    result(j,5) = taxa_acerto; %Acuracia ou porcentagem de acertos 
    result(j,6) = fp; %False Positive
    result(j,7) = fn; %False Negative
    result(j,8) = tp; %True Positive
    result(j,9 )= tn; %True Negative
    result(j,10) = time;
    
end

%Resumo dos resultados obtidos
erro_medio = mean(result(:,4))/length(testClass)
acerto_medio = mean(result(:,5))
desv_acerto_medio = std(result(:,5))
fp_medio = mean(result(:,6))
fn_medio = mean(result(:,7))
tp_medio = mean(result(:,8))
tn_medio = mean(result(:,9))
time_medio = mean(result(:,10));


result_final = [gama C erro_medio acerto_medio desv_acerto_medio fp_medio fn_medio tp_medio tn_medio time_medio];

%Apagar variaveis:
clearvars sigma gama C erro_medio acerto_medio desv_acerto_medio fp_medio fn_medio tp_medio tn_medio;
clearvars fn fp tn tp time;
clearvars j k;