
clear;
# Reading input files

trainingData = dlmread("H:/MachineLearning/Assignment/data_sets1/training_set.csv",",",1,0);
validatonData = dlmread("H:/MachineLearning/Assignment/data_sets1/validation_set.csv",",",1,0);
testData = dlmread("H:/MachineLearning/Assignment/data_sets1/test_set.csv",",",1,0);

fid = fopen('H:/MachineLearning/Assignment/data_sets1/training_set.csv','r');
line = fgetl(fid);
fclose(fid);

global attributeNames=strsplit(line,",");
attributeNames(columns(attributeNames)) = [];

global finalTree = cell(0,0);
global finalTree1 = cell(0,0);
global attributesInTree=cell(0,0);

global parentVal1 = 0;
global parentVal0 = 0;


global count=0;


function [gainAttribute,leafNode,index,parentPosInstances,parentNegInstances] = informationGain(subsetData,parentVal1,parentVal0,currentAttributeNames)
  global attributeNames;
  global attributesInTree;

  gainAttribute = cell();
  leafNode = cell();
  maxInfoGain=0;
  #Calculating Entropy Positives = 1 Negatives = 0

  class = subsetData(:,columns(subsetData));
  noOfPositives = sum(class==1);
  noOfNegatives = sum(class==0);
  parentPosInstances = noOfPositives;
  parentNegInstances = noOfNegatives;

  total = noOfNegatives + noOfPositives;
  
    if(noOfPositives >=1 && noOfNegatives ==0)
        leafNode(1,1) = "1";
        return;
    endif
    if(noOfNegatives >=1 && noOfPositives==0)
        leafNode(1,1) = "0";
        return;
    endif
  
  

  if(total != 0 && noOfPositives!=0 && noOfNegatives!=0)
    Entropy = -((noOfPositives/total)*(log2(noOfPositives/total))) - ((noOfNegatives/total)*(log2(noOfNegatives/total)));
    elseif(total != 0 && noOfPositives!=0 && noOfNegatives==0)
    Entropy = -((noOfPositives/total)*(log2(noOfPositives/total)));
    elseif(total != 0 && noOfPositives==0 && noOfNegatives!=0)
    Entropy = - ((noOfNegatives/total)*(log2(noOfNegatives/total)));
    else
    Entropy = 0;
  endif
  
    if(isempty(currentAttributeNames))
      if(noOfPositives > noOfNegatives)
          leafNode(1,1) = "1";
          return;
      endif
      if(noOfNegatives > noOfPositives)
          leafNode(1,1) = "0";
          return;
      endif
    endif
  
  if(Entropy == 0 || Entropy < 0)
    if(noOfPositives >=1)
        leafNode(1,1) = "1";
        return;
    endif
    if(noOfNegatives >=1)
        leafNode(1,1) = "0";
        return;
    endif
    if(noOfPositives==0 && noOfNegatives ==0)
      if(parentVal1 > parentVal0)
        leafNode(1,1) = "1";
        return;
      endif
      if(parentVal0 > parentVal1)
        leafNode(1,1) = "0";
        return;
      endif
      if(parentVal0 == parentVal1)
        leafNode(1,1) = "0";
        return;
      endif
        
    endif
  endif

#Calculating Information Gain
  i=0;
  index=0;
  for i=1:(columns(attributeNames))
    
    currentAttribute = subsetData(:,i);
    if(sum(strcmp(currentAttributeNames,attributeNames(1,i))) == 1)
        count11=0;
        count10=0;
        count01=0;
        count00=0;
        infoGain=0;
        
        for j=1:rows(currentAttribute)
          if (currentAttribute(j,1)==1 && class(j,1)==1)
            count11=count11+1;
            elseif(currentAttribute(j,1)==1 && class(j,1)==0)
            count10=count10+1;
            elseif(currentAttribute(j,1)==0 && class(j,1)==1)
            count01=count01+1;
            elseif(currentAttribute(j,1)==0 && class(j,1)==0)
            count00=count00+1;
          endif
        endfor
       attributeValue1 = count11+count10;
       attributeValue0 = count01+count00; 
       totalPosNeg = attributeValue1 + attributeValue0;

       if(attributeValue1!=0 && count11 != 0 && count10 != 0)
        Entropy1=-((count11/attributeValue1)*(log2((count11/attributeValue1))))-((count10/attributeValue1)*(log2((count10/attributeValue1))));
        elseif(attributeValue1!=0 && count11 != 0 && count10 == 0  )
        Entropy1=-((count11/attributeValue1)*(log2((count11/attributeValue1))));
        elseif(attributeValue1!=0 && count11 == 0 && count10 != 0  )
        Entropy1=-((count10/attributeValue1)*(log2((count10/attributeValue1))));
        else
        Entropy1=0;
       endif
       
       if(attributeValue1!=0 && count01 != 0 && count00 != 0)
        Entropy0=-((count01/attributeValue0)*(log2((count01/attributeValue0))))-((count00/attributeValue0)*(log2((count00/attributeValue0))));
        elseif(attributeValue1!=0 && count01 != 0 && count00 == 0 )
        Entropy0=-((count01/attributeValue0)*(log2((count01/attributeValue0))));
        elseif(attributeValue1!=0 && count01 == 0 && count00 != 0 )
        Entropy0=-((count00/attributeValue0)*(log2((count00/attributeValue0))));
        else
        Entropy0=0;
       endif
       
       if(totalPosNeg != 0)
        infoGain = Entropy - (((attributeValue1/totalPosNeg)*Entropy1) + ((attributeValue0/totalPosNeg)*Entropy0));
        else
        infoGain = Entropy;
       endif
       
       if(infoGain > maxInfoGain)
        maxInfoGain = infoGain;
        index=i;
       endif
        count11=0;
        count10=0;
        count01=0;
        count00=0; 
        attributeValue1=0;
        attributeValue0=0;
        totalPosNeg=0;
        Entropy1=0;
        Entropy0=0;
    endif

  endfor
  gainAttribute = attributeNames(1,index);
  return;
endfunction


function [bestAttribute] = decisionTree(finalTree,trainingData,parentVal1,parentVal0,currentIndex,currentAttributeNames,level)
  global count;
  global attributesInTree;
  global finalTree1;
  left = 1;
  right = 0;
  i=0;
  j=0;
  selectedAttribute = cell();
  leafNode = cell();
  bestAttribute = cell();
  leftAttributeNames=cell();
  rightAttributeNames=cell();

  [selectedAttribute,leafNode,index,parent1Inst,parent0Inst] = informationGain(trainingData,parentVal1,parentVal0,currentAttributeNames);

  if(isempty(leafNode))
    bestAttribute(1,1) = selectedAttribute;
    elseif(isempty(leafNode)==0)
    bestAttribute(1,1) = leafNode;
    bestAttribute(1,1)
    attributesInTree(1,end+1) = bestAttribute(1,1);
      if(currentIndex == 1)
        i=level+1;
        j=level;
        finalTree1(1,level) = leafNode;
      endif
      if(currentIndex == 0)
        j=level+1;
        i=level;
        finalTree1(1,(level)) = leafNode;
      endif
    return
  endif 

  attributesInTree(1,end+1) = bestAttribute(1,1);
  
  if(isempty(finalTree))
    finalTree1(1,end+1) = bestAttribute;
  endif


#left branch = 1 right branch = 0

 count = count + 1;
 
 subset1 = 	trainingData( find( trainingData(:, index) == 1 ),:);
 subset0 = 	trainingData( find( trainingData(:, index) == 0 ),:);
 parentVal0 = parent0Inst;
 parentVal1 = parent1Inst;

    if(currentIndex == 2)
      leftAttributeNames = currentAttributeNames(~strcmp(currentAttributeNames,bestAttribute));
      rightAttributeNames = currentAttributeNames(~strcmp(currentAttributeNames,bestAttribute));
      i=1;
      j=1;
    endif
    if(currentIndex == 1)
      leftAttributeNames = currentAttributeNames(~strcmp(currentAttributeNames,bestAttribute));
      rightAttributeNames = currentAttributeNames(~strcmp(currentAttributeNames,bestAttribute));
      finalTree1(1,level)=bestAttribute;
      i=level;
      j=level;
    endif
    if(currentIndex == 0)
      rightAttributeNames = currentAttributeNames(~strcmp(currentAttributeNames,bestAttribute));
      leftAttributeNames = currentAttributeNames(~strcmp(currentAttributeNames,bestAttribute));
      finalTree1(1,level)=bestAttribute;
      j=level;
      i=level;
    endif
    decisionTree(finalTree1 ,subset1,parentVal1,parentVal0,left,leftAttributeNames,2*i);

    decisionTree(finalTree1,subset0,parentVal1,parentVal0,right,rightAttributeNames,(2*i)+1);
return;
endfunction

tree = cell();
tree = decisionTree(finalTree,trainingData,parentVal1,parentVal0,2,attributeNames,0);

finalTree1











