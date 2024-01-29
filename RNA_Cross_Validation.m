%Algoritmo FAPESP WILLIAM
%RNA com duas camadas

%Limpar dados do Workspace
clear;
clc;
close all;

%Carega dados coletados e tratados
load datasets.mat;

load('cvp_para_RNA.mat')

%datasetBase = datasetCorrente;
%datasetBase = datasetTensao;
%datasetBase = datasetCorrenteTensao;
datasetBase = datasetAceleracao;


%Tratamento do dataset
for(j=1:length(datasetBase))
    if(datasetBase(j,11)~=1)
        datasetBase(j,11)=0;
    end
    if(datasetBase(j,11)==1)
        datasetBase(j,11)=1;
    end
end

dataset = zscore(datasetBase);

%Atributos selecionados
input(:,1:10)=datasetBase(:,1:10);
output(:,1)=datasetBase(:,11);

x = input';
t = output';


for (i=1:5) %Quantidade de conjuntos de teste e treinamento
    
    %trIdx = cvp.training(j);
    %teIdx = cvp.test(j);
    
    %Xtrain = input(trIdx,:);
    %Ytrain = output(trIdx,1);
    
    %Xtest = input(teIdx,:);
    %Ytest = output(teIdx,1);
  
%Variação do número de neurônios da camada escondida de 1 à 20
for(k2=1:20)
for(k1=1:20) %Roda a rede para 1 a 20 camadas escondidas
    for(m=1:1) %Roda a rede para 5 valores aleatórios iniciais
        time=0;
        accuracy=0;
        nerrors=0;
    
    % Choose a Training Function
    % For a list of all training functions type: help nntrain
    % 'trainlm' is usually fastest.
    %'trainbr' takes longer but may be better for challenging problems.
    % 'trainscg' uses less memory. Suitable in low memory situations.
    %trainFcn = 'trainrp';  % Resilient backpropagation.
    %trainFcn = 'traingdm';  % Gradient Descent with Momentun Backpropagation
    %trainFcn = 'trainscg';  % Scaled conjugate gradient backpropagation.+
    %Usei este
    trainFcn = 'trainlm';  % Levenberg-Marquardt backpropagation.
    %trainFcn = 'traingdm';  % Gradient Descent with Momentun Backpropagation    

    % Create a Pattern Recognition Network
    hiddenLayerSize = [k1 k2];
    net = patternnet(hiddenLayerSize);

    
        % Choose Input and Output Pre/Post-Processing Functions
        % For a list of all processing functions type: help nnprocess
        %net.input.processFcns = {'removeconstantrows','mapminmax'};
        %net.output.processFcns = {'removeconstantrows','mapminmax'};
        net.input.processFcns = {'removeconstantrows'};
        net.output.processFcns = {'removeconstantrows'};

        % Setup Division of Data for Training, Validation, Testing
        % For a list of all data division functions type: help nndivide
        
        %net.divideFcn = 'dividerand';  % Divide dados 
        
        net.divideFcn = 'divideind';  % Divide dados 
        
        net.divideParam.trainInd = trainInd_list(:,i);
        %net.divideParam.valInd   = valInd;
        net.divideParam.testInd  = testInd_list(:,i);


        %ANN Parameters (Fator Momentum)
        net.trainParam.epochs = 1000; %Numero de épocas
        %net.trainParam.goal = 0; %Erro final desejado
        %net.trainParam.lr = 0.01; % Taxa de aprendizagem
        net.trainParam.min_grad = 1e-6;         % Precisão
        %net.trainParam.mc=0.9; %Momentum Constant
        %net.trainParam.max_fail=0; %Parada antecipada
    
        %ANN Parameters (SCG)
        
        % Choose a Performance Function
        % For a list of all performance functions type: help nnperformance
        net.performFcn = 'mse';  % Mean Square Error
        %net.performFcn = 'crossentropy';  % Cross-Entropy

        % Choose Plot Functions
        % For a list of all plot functions type: help nnplot
        %net.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
        %'plotconfusion', 'plotroc','plotregression','plotfit'};

        net.trainParam.showWindow = 0;
        
        % Train the Network
                   
            [net,tr] = train(net,x,t);

            % Test the Network
            tic
            y = net(x);
            time=toc;
            e = gsubtract(t,y);
            performance = perform(net,t,y);

            tind = vec2ind(t);
            yind = vec2ind(y);
            percentErrors = sum(tind ~= yind)/numel(tind);

            % Recalculate Training, Validation and Test Performance
            trainTargets = t .* tr.trainMask{1};
            valTargets = t .* tr.valMask{1};
            testTargets = t .* tr.testMask{1};
            trainPerformance = perform(net,trainTargets,y);
            valPerformance = perform(net,valTargets,y);
            testPerformance = perform(net,testTargets,y) ;
            
            %time=tr.time(tr.num_epochs+1); %Tempo de execução da RNA
            [C,CM]=confusion(t(testInd_list(:,1)),y(testInd_list(:,1)));
        
            %Contagem do número de amostras classifcadas incorretamente
            %Verifica-se se o erro entre target e saída da RNA >0.5
            %numberoferrors=0;
     
        %Criação da Matriz para análise dos resultados
        result(i,k2,k1,m,1)=k1; %Neurônios na camada escondida
        result(i,k2,k1,m,2)=C; %Classificações Incorretas [%]        
        result(i,k2,k1,m,3)=((CM(1,1)+CM(2,2))/length(testInd_list(i,:))); %Acurácia ou porcentagem de acertos 
        result(i,k2,k1,m,4)=CM(1,2); %Falso Positivo
        result(i,k2,k1,m,5)=CM(2,1); %Falso Negativo
        result(i,k2,k1,m,6)=CM(2,2); %True Positive
        result(i,k2,k1,m,7)=CM(1,1); %False Negative
        result(i,k2,k1,m,8)=time; % Tempo de Execução
        result(i,k2,k1,m,9)=trainPerformance; %EQM da amostra de treinamento
        result(i,k2,k1,m,10)=valPerformance; %EQM da amostra de validação
        result(i,k2,k1,m,11)=testPerformance; %EQM da amostra de teste
        result(i,k2,k1,m,12)=performance; %EQM Global (Treinamento+Validação+Teste)
        
        result(i,k2,k1,m,13)=tr.num_epochs; %Número de épocas

    end
  
    end
end
i
end

i=1;
for k2=1:20
    for k1=1:20
   
 %Resumo dos resultados obtidos
 result_final(i,1)=k2;
 result_final(i,2)=k1;
 result_final(i,3)=mean(mean(result(:,k2,k1,:,2)));
 result_final(i,4)=(mean(mean(result(:,k2,k1,:,3)))*5/1152);
 acuracia(1:5)=result(1,5,5,:,3);
 acuracia(6:10)=result(2,5,5,:,3);
 acuracia(11:15)=result(3,5,5,:,3);
 acuracia(16:20)=result(4,5,5,:,3);
 acuracia(21:25)=result(5,5,5,:,3);
 result_final(i,5)=std(acuracia);
 result_final(i,6)=mean(mean(result(:,k2,k1,:,4)));
 result_final(i,7)=mean(mean(result(:,k2,k1,:,5)));
 result_final(i,8)=mean(mean(result(:,k2,k1,:,6)));
 result_final(i,9)=mean(mean(result(:,k2,k1,:,7)));
 result_final(i,10)=mean(mean(result(:,k2,k1,:,8)));
 result_final(i,11)=mean(mean(result(:,k2,k1,:,11)));
 result_final(i,12)=mean(mean(result(:,k2,k1,:,13)));
 i=i+1;
end
end

% Deployment
% Change the (false) values to (true) to enable the following code blocks.
% See the help for each generation function for more information.
if (false)
    % Generate MATLAB function for neural network for application
    % deployment in MATLAB scripts or with MATLAB Compiler and Builder
    % tools, or simply to examine the calculations your trained neural
    % network performs.
    genFunction(net,'myNeuralNetworkFunction');
    y = myNeuralNetworkFunction(x);
end
if (false)
    % Generate a matrix-only MATLAB function for neural network code
    % generation with MATLAB Coder tools.
    genFunction(net,'myNeuralNetworkFunction','MatrixOnly','yes');
    y = myNeuralNetworkFunction(x);
end
if (false)
    % Generate a Simulink diagram for simulation or deployment with.
    % Simulink Coder tools.
    gensim(net);
end
