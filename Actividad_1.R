#https://machinelearningmastery.com/compare-models-and-select-the-best-using-the-caret-r-package/
#https://www.rdocumentation.org/packages/caret/versions/6.0-84/topics/confusionMatrix
#https://www.rdocumentation.org/packages/stats/versions/3.6.1/topics/mcnemar.test
#https://www.rdocumentation.org/packages/caret/versions/6.0-84/topics/diff.resamples
#https://cran.r-project.org/web/packages/caret/vignettes/caret.html
#https://machinelearningmastery.com/compare-models-and-select-the-best-using-the-caret-r-package/
#https://stat.ethz.ch/R-manual/R-devel/library/stats/html/t.test.html
#https://www.r-bloggers.com/using-caret-to-compare-models/
#https://dataaspirant.com/2017/01/09/knn-implementation-r-using-caret-package/
#http://rischanlab.github.io/NaiveBayes.html
#https://www.youtube.com/watch?v=tmahDuw0s3g


#0.- Librerias
library(e1071)
library(caret)
library(datasets)

#1.-Carga de datos
data(iris)
head(iris)
names(iris)
x = iris[,-5]
y = iris$Species

#2.- Esquema de Entrenamiento
control <- trainControl(method="repeatedcv", number=2, repeats=5)

#3.- Entrenamiento de Models
#3.1.- Modelo Naive Bayes
set.seed(7)
modelo.NB = train(x,y,'nb',trControl=control)

#3.2.- Modelo k-Vecinos más cercano
set.seed(7)
modelo.KNN = train(x,y,'knn',trControl=control)

#4.-Test 
#4.1.- Test T-Student
resultado <- resamples(list(NB=modelo.NB, KNN=modelo.KNN))

summary(resultado)
bwplot(resultado)
dotplot(resultado)

diferencias <- diff(resultado)
summary(diferencias)

compare_models(modelo.NB, modelo.KNN)

#4.2.- Text McNemar
res.NB <- predict(modelo.NB$finalModel,x)[["posterior"]]
res.KNN <- predict(modelo.KNN$finalModel,x)
vMatrizConf <- c(0,0,0,0)
for (i in 1:nrow(iris)){
  vNamClases <- c("setosa","versicolor","virginica")
  vClase <- iris[i,5] 
  vClaseNB <- iris[i,5]
  vClaseKNN <- iris[i,5]
  vAuxNB <- 0
  vAuxKNN <- 0
  for (j in 1:3){
    if (vAuxNB < res.NB[i,j]){
      vAuxNB <- res.NB[i,j]
      vClaseNB <- names(res.NB[i,j])
    }
    if (vAuxKNN < res.KNN[i,j]){
      vAuxKNN <- res.KNN[i,j]
      vClaseKNN <- names(res.KNN[i,j])
    }
  }
  if (vClase == vClaseNB && vClase == vClaseKNN ){
    vMatrizConf[1] <- vMatrizConf[1] + 1
  }
  if (vClase == vClaseNB && vClase != vClaseKNN ){
    vMatrizConf[2] <- vMatrizConf[2] + 1
  }
  if (vClase != vClaseNB && vClase == vClaseKNN ){
    vMatrizConf[3] <- vMatrizConf[3] + 1
  }
  if (vClase != vClaseNB && vClase == vClaseKNN ){
    vMatrizConf[4] <- vMatrizConf[4] + 1
  }
}

Performance <-
  matrix(vMatrizConf,
         nrow = 2,
         dimnames = list("Modelo NB" = c("Correcto", "Incorrecto"),
                         "Modelo KNN" = c("Correcto", "Incorrecto")))
Performance
mcnemar.test(Performance)

