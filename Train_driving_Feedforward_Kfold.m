%NEWPLOT Prepares figure, axes for graphics according to NextPlot.
% script to apply training to the Risk dataset with a feedforward neural network
% 
% Francisco Costela 2019
% clear all
rng('shuffle'); %Initialization of the random seed

path = '/Users/FranciscoCostela/Desktop/Risk/Sessions';
matrix_train= []; 
matrix_validate = [];

% labels for eye features
myenums = { 'id','trial','vidnum','nsacs','magsac','nusacs','magusacs','nfixs','fixdur','disperx','dispery','nblinks','pupincrease','risk'};
enums = {};
for i =1:14
    enums.(myenums{i}) = i;    
end

load riskdb

% Get rows with risk detection measures
yes = riskdb(riskdb(:,14)==1,:);
no = riskdb(riskdb(:,14)==0,:);

% Use 90% for training, 10% for validation
mysizeyes = size(yes,1);
trainsizeyes = round(mysizeyes*0.9);
mysizeno = size(no,1);
trainsizeno = round(mysizeno*0.9);

%10-fold cross validation
for k=1:10
    
    % randomly fill values in each cross validation
    random_yes = yes(randperm(size(yes, 1)), :);
    random_no = no(randperm(size(no, 1)), :);
    
    % create matrix for training - validation
    matrix_train =  [random_yes(1:trainsizeyes-1,:) ; random_no(1:trainsizeno-1,:)];    
    matrix_validate =  [random_yes(trainsizeyes:end,:) ; random_no(trainsizeno:end,:)];
    
    matrix_train = matrix_train(randperm(size(matrix_train, 1)), :);
    matrix_validate = matrix_validate(randperm(size(matrix_validate, 1)), :);
    
    %itrain is the number of models
    for itrain=1:7 
        
        disp([ 'Training model number ' num2str(itrain)]);
        
        input_matrix=[]; %nan*ones( maxlength*5, length(matrix_train)); % Input of the Netword
        target_matrix=[]; %nan*ones( maxlength, length(matrix_train));  %Desirable output of the network
        
        %     for ii=1:1:length(matrix_train)
        
        switch itrain
            case 1
                % all
                for j=1:length(matrix_train)
                    input_matrix(:,j) = [matrix_train(j,enums.magsac); matrix_train(j,enums.nfixs); matrix_train(j,enums.fixdur); matrix_train(j,enums.disperx); matrix_train(j,enums.dispery); matrix_train(j,enums.nblinks)];
                    target_matrix(:,j) = matrix_train(j,enums.risk);
                end
            case 2
                % all no pupil
                for j=1:length(matrix_train)
                    input_matrix(:,j) = [matrix_train(j,enums.magsac),matrix_train(j,enums.fixdur), matrix_train(j,enums.nblinks)];
                    target_matrix(:,j) = matrix_train(j,enums.risk);
                end
                
            case 3 % saccades, fixation duration, disperxion x, dispersion y
                for j=1:length(matrix_train)
                    input_matrix(:,j) = [matrix_train(j,enums.magsac), matrix_train(j,enums.fixdur), matrix_train(j,enums.disperx), matrix_train(j,enums.dispery) ];
                    target_matrix(:,j) = matrix_train(j,enums.risk);
                end
                
            case 4 % just fixational eye movement
                for j=1:length(matrix_train)
                    input_matrix(:,j) = [matrix_train(j,enums.magsac), matrix_train(j,enums.disperx)];
                    target_matrix(:,j) = matrix_train(j,enums.risk);
                end
                
            case 5 % just saccades/microsaccades and drift
                for j=1:length(matrix_train)
                    input_matrix(:,j) = [matrix_train(j,enums.magsac), matrix_train(j,enums.fixdur), matrix_train(j,enums.disperx) ];
                    target_matrix(:,j) = matrix_train(j,enums.risk);
                end
                
            case 6 % just saccades/microsaccades and drift
                for j=1:length(matrix_train)
                    input_matrix(:,j) = [matrix_train(j,enums.magsac), matrix_train(j,enums.fixdur), matrix_train(j,enums.disperx), matrix_train(j,enums.pupincrease) ];
                    target_matrix(:,j) = matrix_train(j,enums.risk);
                end
            case 7
                % all
                for j=1:length(matrix_train)
                    input_matrix(:,j) = [matrix_train(j,enums.magsac); matrix_train(j,enums.nfixs); matrix_train(j,enums.fixdur); matrix_train(j,enums.disperx); matrix_train(j,enums.dispery); matrix_train(j,enums.nblinks);matrix_train(j,enums.pupincrease)];
                    target_matrix(:,j) = matrix_train(j,enums.risk);
                end
        end
        
        % Define feedforward neural network
        net = feedforwardnet([32 32]);
        
        step=1;
        
        net.input.processFcns={'removeconstantrows','mapstd'}; %parameterers of the network training
        
        a = tic;
        [net,tr] = train(net,input_matrix(:,1:step:end),target_matrix(:,1:step:end)); %training. Note that step help to reduce the number of samples in case we have more than 90000.
        disp(toc(a));
        
        %The next parameterers define our trained network
        Iw=net.IW; %input weight
        b=net.b; %bias
        LW=net.LW; %Layer weights
        norm_param_in=net.input.processSettings{1}; %input normalization parameteers
        norm_param_out=net.output.processSettings{1}; %output normalization parameteers
        
        %Now we save our models into .mat files
        fname=['model_32x32_itrain_',num2str(itrain),'.mat'];
        save(fname,'Iw','b','LW','norm_param_in','norm_param_out') %You dont need the neural network toolbox to use the parameteers
        fname=['net_32x32_itrain_',num2str(itrain),'.mat'];
        save(fname,'net','input_matrix'); 
        %        
        
        input_matrix=[]; % Input of the Netword
        target_matrix=[]; %Desirable output of the network
                
        disp('validating');
        
        switch itrain
            case 1
                % all
                for j=1:length(matrix_validate)
                    input_matrix(:,j) = [matrix_validate(j,enums.magsac); matrix_validate(j,enums.nfixs); matrix_validate(j,enums.fixdur); matrix_validate(j,enums.disperx); matrix_validate(j,enums.dispery); matrix_validate(j,enums.nblinks)];
                    target_matrix(:,j) = matrix_validate(j,enums.risk);
                end
            case 2
                % all no pupil
                for j=1:length(matrix_validate)
                    input_matrix(:,j) = [matrix_validate(j,enums.magsac),matrix_validate(j,enums.fixdur), matrix_validate(j,enums.nblinks)];
                    target_matrix(:,j) = matrix_validate(j,enums.risk);
                end
                
            case 3 % saccades, fixation duration, disperxion x, dispersion y
                for j=1:length(matrix_validate)
                    input_matrix(:,j) = [matrix_validate(j,enums.magsac), matrix_validate(j,enums.fixdur), matrix_validate(j,enums.disperx), matrix_validate(j,enums.dispery) ];
                    target_matrix(:,j) = matrix_validate(j,enums.risk);
                end
                
            case 4 % just fixational eye movement
                for j=1:length(matrix_validate)
                    input_matrix(:,j) = [matrix_validate(j,enums.magsac), matrix_validate(j,enums.disperx)];
                    target_matrix(:,j) = matrix_validate(j,enums.risk);
                end
                
            case 5 % just saccades/microsaccades and drift
                for j=1:length(matrix_validate)
                    input_matrix(:,j) = [matrix_validate(j,enums.magsac), matrix_validate(j,enums.fixdur), matrix_validate(j,enums.disperx) ];
                    target_matrix(:,j) = matrix_validate(j,enums.risk);
                end
                
            case 6 % just saccades/microsaccades and drift
                for j=1:length(matrix_validate)
                    input_matrix(:,j) = [matrix_validate(j,enums.magsac), matrix_validate(j,enums.fixdur), matrix_validate(j,enums.disperx), matrix_validate(j,enums.pupincrease) ];
                    target_matrix(:,j) = matrix_validate(j,enums.risk);
                end
            case 7
                % all
                for j=1:length(matrix_validate)
                    input_matrix(:,j) = [matrix_validate(j,enums.magsac); matrix_validate(j,enums.nfixs); matrix_validate(j,enums.fixdur); matrix_validate(j,enums.disperx); matrix_validate(j,enums.dispery); matrix_validate(j,enums.nblinks);matrix_validate(j,enums.pupincrease) ];
                    target_matrix(:,j) = matrix_validate(j,enums.risk);
                end
        end
        
        %validation of the network
        testI=input_matrix;%(:,end-validation_samples:end);
        testT=target_matrix;%(:,end-validation_samples:end);
        testY(itrain,:) = net(testI);
        avg_error(k,itrain)=nanmean(abs(testT-testY(itrain,:)));
        
        values = double(testY(itrain,:)>0.5);
        
        c = confusionmat(target_matrix,values);
        
        accuracy(k,itrain) = (c(1,1)+c(2,2))/length(target_matrix);
        
        fprintf(['Mean Error:',num2str(avg_error(itrain)),'\n']);
        name = [ 'Result' int2str(k) '-Fold-Model' int2str(itrain) ];
        save(name, 'avg_error', 'k','itrain','input_matrix','target_matrix','testY', 'accuracy');
    end
    
end
disp(accuracy);


