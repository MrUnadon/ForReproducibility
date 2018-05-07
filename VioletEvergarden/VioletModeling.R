invisible({rm(list=ls());gc();gc()})
##########################################################################
# Tokyo.R 2018-06-09 ベイズ・統計モデリング
# ヴァイオレット・エヴァーガーデンのベイズ統計モデリング
# Author: MrUnadon, Date: 2018-05-06
##########################################################################

# libraries
library(RCurl)
library(rlist)
library(rjson)
library(RcppRoll)
library(tidyverse)
library(anytime)
library(formattable)
library(scales)
library(shinystan)
library(ggmcmc)
library(rstan)
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)


# Index ------------------------------------------------------------------
# 1. AnimeAPIからのTwitter公式アカウントフォロワー数取得
# 2. ローデータの可視化
# 3. Stanに渡すデータの準備
# 4. 結果の取り出しと整理
# 5. 結果の可視化1(トレンド)
# 6. 結果の可視化2(主要パラメータ)
# ------------------------------------------------------------------------




# 1. AnimeAPIからのTwitter公式アカウントフォロワー数取得 -----------------

# Referrence
  # http://izunyan.hatenablog.com/
  # https://github.com/Project-ShangriLa/sora-playframework-scala

#アニメの放送年:ベクトル(複数指定可)
yearsBC<-c(2018)
#アニメシーズン:ベクトル（1=冬、2=春、3=夏、4=秋: 複数指定可能）
seasons<-c(1)
#現在から何時間前までのデータを取得するか
endTime<-c(1)
#50時間 ✕ 何回分のデータを取得するか
samplings<-c(56)
  # as.POSIXct("2018-05-06") - as.POSIXct("2018-01-08") # データ取得日(5月6日)から、APIデータ記録開始の1月8日まで
  # 118 * 24 / 50 # 50時間を1単位で何回取ればよいか

# Masterデータを取得し、Twitter公式アカウント名とアニメ名を取り出す
master<-data.frame(matrix(NA,0,4))
for(i in 1: length(yearsBC)){
  for(j in 1: length(seasons)){
    tmpFull <- paste("http://api.moemoe.tokyo/anime/v1/master",yearsBC[i],seasons[j],sep = "/") %>% 
      sprintf("Array") %>%
      list.load() %>% list.ungroup() 
    N_anime<-length(tmpFull)/15
    for (k in 1:N_anime){
      tmpTwAcName<-unlist(tmpFull[c((k*15)-13)][1])
      tmpAnime<-unlist(tmpFull[c((k*15)-11)][1])
      master[(length(master[,1])+1),]<-c(yearsBC[i],j,tmpTwAcName,tmpAnime)
    }
  }
}

# マスター情報の確認
formattable(master)

# endpointの決定
subtract_UNIXtime<-c()
for(t in 1:samplings){subtract_UNIXtime[t]<-c((((60*30*100)*t)-(60*30*100))-3)}
end<-c(as.numeric(Sys.time())-c(endTime*60))
endP<-round(c(end-subtract_UNIXtime)-180000,0)

# Follower Historyの抽出(ヴァイオレット・エヴァーガーデン)
FollowerHistory<-data.frame(matrix(NA,0,5))
for(t in 1: samplings){
  tmpRes <- paste("http://api.moemoe.tokyo/anime/v1/twitter/follower/history?account=Violet_Letter&end_date=",endP[t],sep="") %>%
    sprintf("account") %>%
    list.load("") %>% list.ungroup()
  for(u in 1 : (length(tmpRes) / 2 )){
    FollowerHistory[(length(FollowerHistory$X1)+1),]<-c(2018,1,"ヴァイオレットエヴァーガーデン",as.numeric(unlist(tmpRes[[2*u]][1])),as.numeric(unlist(tmpRes[[(2*u)-1]][1])))
  }
}


#データの列名や時間情報を整理
base <- data.frame(Date = as.Date(seq(as.POSIXct("2018-01-09"), as.POSIXct("2018-05-07"), by = "days")))
TidyAnime<- FollowerHistory %>% 
  dplyr::mutate(Year = as.integer(X1),
                Season = as.integer(X2),
                Anime = as.factor(X3),
                UnixTime = as.integer(X4),
                Follower = as.integer(X5)) %>%
  dplyr::mutate(Time = as.POSIXct(as.Date(anytime(as.numeric(UnixTime),tz = "Asia/Tokyo"))),
                Date = lubridate::date(Time)
  ) %>%
  dplyr::select(Year, Season, Anime, Time, Date, Follower) %>%
  dplyr::group_by(Date) %>%
  dplyr::arrange(desc(Time), .by_group = TRUE) %>%
  dplyr::distinct(Date, .keep_all = TRUE) %>%
  dplyr::ungroup() %>%
  dplyr::right_join(base) %>%
  dplyr::full_join(data.frame(Date = lubridate::date(as.POSIXct(c("2018-01-09"))) + (seq(1:13) * 7 - 6),
                              OnAir = rep(1),
                              OnAirDate = lubridate::date(as.POSIXct(c("2018-01-09"))) + (seq(1:13) * 7 - 6))
  ) %>%
  dplyr::arrange(Date) %>%
  tidyr::replace_na(list(OnAir = 0)) %>%
  tidyr::fill(Year, Anime, Season, .direction = "down") %>%
  dplyr::filter(Date > as.POSIXct("2018-01-08") & Date < as.POSIXct("2018-04-18"))%>%
  dplyr::mutate(OnAir = ifelse(OnAir == 1, "放送日", "非放送日")) %>%
  as.data.frame()


#データの確認
formattable(TidyAnime)

# .Rdataで保存
save(TidyAnime, file = "data/TidyAnime.Rdata")
load(file = "data/TidyAnime.Rdata")


# 2. ローデータの可視化 --------------------------------------------------

# フォロワー数のトレンド
g1 <- ggplot(TidyAnime) + 
  theme_bw(base_size = 12, base_family = "HiraKakuProN-W6") +
  geom_line(aes(x = Time, y = Follower), colour = "darkgray") +
  geom_point(aes(x = Time, y = Follower)) +
  scale_x_datetime(expand = c(0.01, 0.01), breaks = scales::date_breaks("7 days")) +
  theme(axis.text = element_text(hjust = 1, angle = 45)) +
  scale_colour_manual(values = c("#6495ED", "#CD2626")) +
  labs(title = "ヴァイオレット・エヴァーガーデンのフォロワー数推移",
       subtitle = "放送開始1月10日-放送修了4月4日",
       x = "Date") +
  geom_vline(xintercept = c(as.POSIXct("2018-01-10"), as.POSIXct("2018-04-04")), linetype = 2)
# 描画
print(g1)

# 保存
ggsave(g1, filename = "figure/01_FollwerTrend.png",
       w = 9, h = 4, dpi = 400 )


# フォロワー増加量のトレンド
delta_df <- TidyAnime %>% #増加量を追加して、ラグによる欠損が出た1月9日を除外
  dplyr::mutate(lag_Follower = dplyr::lag(Follower))%>%
  dplyr::mutate(delta = Follower - lag_Follower)%>%
  dplyr::filter(Date > as.POSIXct("2018-01-09"))


g2  <- ggplot(delta_df) + 
  theme_classic(base_size = 8,base_family = "HiraKakuProN-W6")+
  geom_vline(aes(xintercept = as.POSIXct(OnAirDate)), linetype = 1, colour = "gray")+ 
  geom_vline(aes(xintercept = as.POSIXct(as.Date(OnAirDate) + 1)), linetype = 2, colour = "gray")+ 
  geom_hline(yintercept = 0,size=0.3,linetype=2,colour="gray50")+
  geom_line(aes(x=as.POSIXct(Time),y=delta), colour = "gray20")+
  geom_point(aes(x=as.POSIXct(Time),y=delta, colour = OnAir))+
  scale_x_datetime(expand = c(0.01, 0.01), breaks = scales::date_breaks("7 days"))+
  theme(axis.text = element_text(hjust = 1, angle = 45)) +
  scale_colour_manual(values = c("#6495ED", "#CD2626")) +
  labs(title = "ヴァイオレット・エヴァーガーデンのフォロワー増加数推移",
       subtitle = "実線 = 放送日, 破線 = 放送翌日",
       x = "Date") + 
  theme(legend.position = "bottom", legend.direction = "horizontal") 

# 描画
print(g2)

# 保存
ggsave(g2, filename = "figure/02_FollwerTrend2.png",
       w = 9, h = 4, dpi = 400 )


# 3. Stanに渡すデータの準備 ----------------------------------------------
# Stan用データ整形1(Long形式で渡すデータ用)
stan_base <- delta_df %>%
  dplyr::mutate(
    TimePoint = 1:nrow(delta_df),
    stan_delta = ifelse(is.na(delta), -9999, delta),
    Term = c(rep(c(1:13), each = 7),
             rep(13, nrow(delta_df) - length(c(rep(c(1:13), each = 7))))
    ) 
  ) %>%
  dplyr::group_by(Term) %>%
  dplyr::mutate(
    TermPoint = row_number(),
    TermBase = ifelse(TermPoint == 1, stan_delta, NA)
  ) %>%
  dplyr::ungroup() %>%
  tidyr::fill(TermBase, .direction = "down")

# stan用に作った列のあるデータ
formattable(stan_base)

# stan用データ整形2(図の赤点のみに相当するデータ用)
Term_dat <- stan_base %>%
  dplyr::distinct(Term, .keep_all = TRUE) 


# 必要な情報だけ明示的に抽出
TimeLength <- max(stan_base$TimePoint)
TermLength <- nrow(Term_dat)
TermPoint <- stan_base$TermPoint
TermIndex <- stan_base$Term
TermNo <- Term_dat$Term
TermBase <- Term_dat$TermBase
Y <- stan_base$stan_delta
each_NumNA <- sum(is.na(stan_base$delta))
base_NumNA <- sum(Term_dat$TermBase == -9999)

# Stan用にリスト型でデータを整理
datastan <- list(
  TimeLength = TimeLength,
  TermLength = TermLength,
  TermPoint = TermPoint,
  Y = Y,
  TermIndex = TermIndex,
  TermNo = TermNo,
  TermBase = TermBase,
  base_NumNA = base_NumNA,
  each_NumNA = each_NumNA
  )

# モデルのコンパイルとrds形式での保存
VioletModel <- stan_model("stan/model/ForVioletEvergarden.stan")

# サンプリングの実行
fit <- sampling(VioletModel,
                data = datastan,
                seed = 1234,
                iter = 2000,
                warmup = 1000
                )

# 結果の保存
  # save(fit, file = "stan/fit/VioletModel_fit.Rdata")

# 結果の読み込み
load("stan/fit/VioletModel_fit.Rdata")

# 収束診断(確認済み)
  # launch_shinystan(fit)



# 4. 結果の取り出しと整理 ------------------------------------------------
# MCMCサンプルの取り出し & データフレーム化
samples <- ggmcmc::ggs(fit)

# トレンドの期待値のみ取り出して95%信用区間を計算
each_mu <- samples %>%
  dplyr::filter(str_detect(Parameter, "each_mu")) %>%
  dplyr::group_by(Parameter) %>%
  dplyr::summarise(EAP = mean(value),
                   lower = quantile(value, 0.025),
                   upper = quantile(value, 0.975)
  ) %>%
  dplyr::ungroup() 

# 元データと結合
df_res <- cbind(stan_base, each_mu)


# 5. 結果の可視化1(トレンド) ---------------------------------------------

# トレンドの可視化
g3 <- ggplot(df_res, aes(x = as.POSIXct(as.Date(Time)))) +
  theme_classic(base_size = 8,base_family = "HiraKakuProN-W6")+
  geom_vline(aes(xintercept = as.POSIXct(OnAirDate)), linetype = 1, colour = "gray")+ 
  geom_vline(aes(xintercept = as.POSIXct(as.Date(OnAirDate) + 1)), linetype = 2, colour = "gray")+ 
  geom_hline(yintercept = 0,size=0.3,linetype=2,colour="gray50")+
  geom_line(aes(y=delta), colour = "darkgray")+
  geom_point(aes(y=delta, colour = factor(OnAir)))+
  geom_ribbon(aes(ymax = upper, ymin = lower), alpha = 0.7) +
  geom_line(aes(y = EAP), alpha = 1, size = 0.2) +
  scale_x_datetime(expand = c(0.01, 0.01), breaks = scales::date_breaks("7 days"))+
  theme(axis.text = element_text(hjust = 1, angle = 45)) +
  scale_colour_manual(values = c("#6495ED", "#CD2626")) +
  labs(title = "ヴァイオレット・エヴァーガーデンのフォロワー増加数推移と予測",
       subtitle = "実線 = 放送日, 破線 = 放送翌日, 帯 = 推定平均値の95%信用区間",
       x = "Date") + 
  theme(legend.position = "bottom", legend.direction = "horizontal") 


# 描画
print(g3)

# 保存
ggsave(g3, filename = "figure/03_EstimatedTrend.png",
       w = 9, h = 4, dpi = 400 )



# ベーストレンドの期待値のみ取り出して95%信用区間を計算
base_mu <- samples %>%
  dplyr::filter(str_detect(Parameter, "base_mu")) %>%
  dplyr::group_by(Parameter) %>%
  dplyr::summarise(EAP = mean(value),
                   lower = quantile(value, 0.025),
                   upper = quantile(value, 0.975)
  ) %>%
  dplyr::ungroup() %>%
  dplyr::mutate(Term = 1:13,
                TermBase = ifelse(TermBase == -9999, NA, TermBase))

g4 <- ggplot(base_mu, aes(x = Term, y = EAP)) +
  theme_bw(base_size = 12, base_family = "HiraKakuProN-W6")+
  geom_ribbon(aes(ymax = upper, ymin = lower), alpha = 0.2, fill = "darkred") +
  geom_hline(yintercept = 0, size=0.3,linetype=2,colour="gray50")+
  geom_line(colour = "darkred") +
  geom_point(aes(y = TermBase), colour = "darkgray") +
  scale_x_continuous(expand = c(0.01, 0.01)) +
  labs(title = "ヴァイオレット・エヴァーガーデンの放送日フォロワー増加数の推移と予測",
       subtitle = "黒点 = 放送日フォロワー増加数, 実線 = EAP, 帯 = 推定値の95%信用区間",
       x = "Term") + 
  theme(legend.position = "bottom", legend.direction = "horizontal") 

# 描画
print(g4)

# 保存
ggsave(g4, filename = "figure/04_EstimatedBaseTrend.png",
       w = 9, h = 4, dpi = 400 )


# 6. 結果の可視化2(主要パラメータ) ---------------------------------------

# 全体の減衰係数, 各タームでの減衰係数, 最終回増加数
pars <- samples %>%
  dplyr::filter(str_detect(Parameter, "each_gamma") |
                str_detect(Parameter, "base_gamma") |
                str_detect(Parameter, "R") |
                Parameter == "r"
                ) %>%
  dplyr::select(Parameter, value)

# 可視化
g5 <- ggplot(pars) +
  geom_histogram(mapping = aes(x = value, y = ..density.., fill = Parameter, colour = Parameter), bins = 50, alpha = 0.7) +
  geom_density(mapping = aes(x = value), alpha = 0.3, colour = "black") +
  facet_wrap(~Parameter, scales = "free") +
  theme_bw(base_size = 12, base_family = "HiraKakuProN-W6") +
  scale_fill_manual(values = c("#00008B", "#9ACD32", "#006400", "#5D478B"))+
  scale_colour_manual(values = c("#00008B", "#9ACD32", "#006400", "#5D478B"))

# 描画
print(g5)

# 保存
ggsave(g5, filename = "figure/05_EstimatedParameters.png",
       w = 9, h = 4, dpi = 400 )


# パラメータの推定値
summary(fit)$summary[, c(1,4,8,9,10)] %>%
  data.frame() %>%
  round(3) %>%
  formattable()

# Fin
