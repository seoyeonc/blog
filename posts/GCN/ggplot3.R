# IRdisplay 패키지 로드
library(IRdisplay)
# 그래프 출력 크기 설정
options(jupyter.plot_scale=4)
options(repr.plot.width=6,repr.plot.height=4,repr.plot.res=300)
library(tidyverse)
library(patchwork)

figsize <- function(width=6,height=4){
    options(repr.plot.width=width,repr.plot.height=height,repr.plot.res=300)
}

ggplot <- function(...){
    ggplot2::ggplot(...)+      
    theme_bw()+theme(panel.border=element_blank(),axis.line=element_line(colour="black"))+
    theme(axis.title.x=element_text(size=rel(1),lineheight=0.9,face="bold.italic"))+
    theme(axis.title.y=element_text(size=rel(1),lineheight=0.9,face="bold.italic"))+
    theme(plot.title=element_text(size=rel(2),lineheight=0.9,face="bold.italic"))+
    theme(plot.margin = unit(c(3,3,0,0), "mm"))
}

prepare_data <- function(x, y = NULL) {
  if (is.null(y)) {
    y=x 
    if (!is.vector(y)) {
      x = 1:dim(y)[1]
    } else {
      x = 1:length(y)
    }    
  }
  if (!is.vector(y)) {
      dfx = data.frame(x)
      dfy = data.frame(y)
      df = cbind(dfx,dfy) 
      df = pivot_longer(df,cols = colnames(y), names_to = "label", values_to = "y")        
  } else {
      df = data.frame(x=x,y=y)
  }
  return(df)
}

## main geoms

line <- function(x, y=NULL,label=NULL, ...) {
  df = prepare_data(x, y)
  return(geom_line(data=df, mapping=aes(x = x, y = y, col=label), ...))
}

point <- function(x, y=NULL,label=NULL, ...) {
  df = prepare_data(x, y)
  return(geom_point(data=df, mapping=aes(x = x, y = y, col=label), ...))
}

## 2d geoms 

smooth <- function(x, y=NULL,label=NULL, ...) {
  df = prepare_data(x, y)
  return(geom_smooth(data=df, mapping=aes(x = x, y = y, col=label), ...))
}

area <- function(x, y=NULL,label=NULL, ...) {
  df = prepare_data(x, y)
  return(geom_area(data=df, mapping=aes(x = x, y = y, fill=label,col=label), alpha=0.1,...))
}

step <- function(x, y=NULL,label=NULL, ...) {
  df = prepare_data(x, y)
  return(geom_step(data=df, mapping=aes(x = x, y = y,col=label),...))
}

jitter <- function(x, y=NULL,label=NULL, ...) {
  df = prepare_data(x, y)
  return(geom_jitter(data=df, mapping=aes(x = x, y = y,col=label),...))
}

## 1d geoms
histogram <- function(y,label=NULL, ...) {
  df = prepare_data(y)
  return(geom_histogram(data=df, mapping=aes(x = y, y=stat(density),fill=label),alpha=0.5, bins=30,position = "identity",...))
}

density <- function(y,label=NULL, ...) {
  df = prepare_data(y)
  return(geom_density(data=df, mapping=aes(x = y, fill=label,col=label),alpha=0.25,...))
}

qq <- function(y,label=NULL, ...) {
  df = data.frame(y)
  return(geom_qq(data=df, mapping=aes(sample = y,col=label),...))
}

qq_line <- function(y,label=NULL, ...) {
  df = data.frame(y)
  return(geom_qq_line(data=df, mapping=aes(sample = y,col=label),...))
}

## compare geoms 

col <- function(x, y=NULL,label=NULL, ...) {
  df = prepare_data(x, y)
  df$x = as.factor(df$x)
  return(geom_col(data=df, mapping=aes(x = x, y = y, fill=label),position='dodge', ...))
}

boxplot <- function(x, y=NULL,label=NULL, ...) {
  if (is.null(y)) {
    y=x 
    x=0
  }
  df = prepare_data(x, y)
  return(geom_boxplot(data=df, mapping=aes(x = x, y = y, col=label), ...))
}

violin <- function(x, y=NULL,label=NULL, ...) {
  if (is.null(y)) {
    y=x 
    x=0
  }
  df = prepare_data(x, y)
  return(geom_violin(data=df, mapping=aes(x = x, y = y, fill=label, color=label),alpha=0.5, scale='area', ...))
}