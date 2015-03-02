#Implementation of Naive Bayes algorithm 

clear;
inputFileName="reuters.mat";

#Reading input
load(inputFileName);

trainyWithClass = [train trainy];
noOfClasses = rows(unique(trainy));
classes = sort(unique(trainy),'descend');
prior=cell();
vocabulary=word_indices;

#Train NB
for c = 1:noOfClasses
  noDocsInClass=0;
  termCount=[];
  subsetTrainyWithClass=[];
  subsetTrainyWithOutClass=[];

  noDocsInClass = rows(trainy(trainy==classes(c,1),:));
  prior(c,1) = (noDocsInClass/rows(train));
  
  subsetTrainyWithClass = trainyWithClass((trainyWithClass(:,columns(trainyWithClass))==classes(c,1)),:);
  subsetTrainyWithClass(:,end)=[];
  subsetTrainyWithOutClass = subsetTrainyWithClass;
  
  for term = 1:rows(vocabulary)
   termCount(term,1) = sum(subsetTrainyWithOutClass(:,term));
  end
  
  for term = 1:rows(vocabulary)
    conditionalProbability(term,c) = ((termCount(term,1)+1)/((sum(subsetTrainyWithOutClass(:)))+ columns(train)));
  end
  
end

#NB - Testing 
for d = 1:rows(test)
  currentDoc = test(d,:);
  for c = 1:noOfClasses
    score(c,1) = log10(prior{c,1});
    for term = 1:columns(test)
      score(c,1) = score(c,1) + (currentDoc(1,term)*(log10(conditionalProbability(term,c))));
    end
  end
  
  index = find(score == max(score(:)));
  argMax=classes(index);
  docClass(d,1)=argMax;
  
  score=[];
  currentDoc=[];
  argMax=[];
  index=[];
end

#Finding Accuracy
classesMatched =sum(docClass==testy);
accuracy = classesMatched/rows(test);

printf("\n No of documents classified correctly - %d \n",classesMatched);
printf("\nAccuracy = %f\n",accuracy);
