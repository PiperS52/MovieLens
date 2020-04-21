#Capstone Movielens
install.packages("data.table")
library(data.table)
library(tidyverse)
library(caret)
library(dplyr)
library(stringr)

### creating edx and validation datasets ###
# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

###

######      Overview of Data      ######
# useful summary stats
library(lubridate)
library(dslabs)

#This is a plot of average rating for each week against date. Shows no clear sign of a time trend
# within the data and thus decide to omit from my model.
edx <- mutate(edx, date = as_datetime(timestamp), date = as.Date(date))
edx %>% mutate(date = round_date(date, unit = "week")) %>%
  group_by(date) %>%
  summarize(rating = mean(rating)) %>%
  ggplot(aes(date, rating)) +
  geom_point() +
  geom_smooth()

#Looking at genres. Genre categories with >1000 ratings. Plot shows average rating for each
#genre category
edx %>% group_by(genres) %>%
  summarize(n = n(), avg = mean(rating), se = sd(rating)/sqrt(n())) %>%
  filter(n >= 100000) %>% 
  mutate(genres = reorder(genres, avg)) %>%
  ggplot(aes(x = genres, y = avg, ymin = avg - 2*se, ymax = avg + 2*se)) + 
  geom_point() +
  geom_errorbar() + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

#number of users and movies
edx %>% as_tibble()
edx %>% summarize(n_users = n_distinct(userId), 
                  n_movies = n_distinct(movieId))
#Not every user rated every movie
#Some movies get rated more than others. Some users are more active at rating.
# distributions, (fig.hold='hold'), echo=FALSE,{out.width="50%"}
edx %>% 
  dplyr::count(movieId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() + 
  ggtitle("Movies")
edx %>%
  dplyr::count(userId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() +
  ggtitle("Users")



######        Method/analysis        ######
options(pillar.signif = 9)
library(dplyr)
library(caret)
set.seed(755, sample.kind = "Rounding")
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.2, list = FALSE)
train_set <- edx[-test_index,]
test_set <- edx[test_index,]
test_set <- test_set %>% semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")
RMSE <- function(true_ratings,predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}
#model 1 - just the average
mu <- mean(train_set$rating)
mu
#this yields the following RMSE
naive_rmse <- RMSE(mu, test_set$rating)
naive_rmse
#table
rmse_results <- tibble(method = "Just the average", RMSE = naive_rmse)
print.data.frame(rmse_results)

#model 2 - movie effects
movie_avgs <- train_set %>% group_by(movieId) %>% summarize(b_i = mean(rating - mu))
movie_avgs
?qplot
qplot(b_i, data = movie_avgs, bins = 10, colour = I("black"))
  #to get the RMSE
predicted_ratings <- mu + test_set %>% left_join(movie_avgs, by = "movieId") %>%
  pull(b_i)
predicted_ratings
movie_rmse <- RMSE(predicted_ratings, test_set$rating)
movie_rmse
rmse_results <- bind_rows(rmse_results,
                          tibble(method = "Movie Effect Model",
                                 RMSE = movie_rmse))
print.data.frame(rmse_results)

#model 3 - movie and user effects
#avg rating for users who have rated >100 movies
train_set %>% 
  group_by(userId) %>% 
  summarize(b_u = mean(rating)) %>% 
  filter(n()>=100) %>%
  ggplot(aes(b_u)) + 
  geom_histogram(bins = 30, color = "black")

user_avgs <- train_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

predicted_ratings <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

user_movie_rmse <- RMSE(predicted_ratings, test_set$rating)

rmse_results <- bind_rows(rmse_results,
                          tibble(method = "User & Movie Effect Model", RMSE = user_movie_rmse))


print.data.frame(rmse_results)

#model 4 - movie + user + genre
genres_avgs <- train_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by="userId") %>%
  group_by(genres) %>%
  summarize(b_g = mean(rating - mu - b_i - b_u))

predicted_ratings <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genres_avgs, by="genres") %>%
  mutate(pred = mu + b_i + b_u + b_g) %>%
  pull(pred)

user_movie_genres_rmse <- RMSE(predicted_ratings, test_set$rating)

rmse_results <- bind_rows(rmse_results,
                          tibble(method = "Movie & User & Genres Effect Model",
                                 RMSE = user_movie_genres_rmse))

print.data.frame(rmse_results)
#regularisation
#according to the b_i estimate, all the top 10 and worst 10 are rather 
#obscure movies. The supposed best and worst films were rated by very
#few users. This means more uncertainty and noisy estimates.

#model 5 - reg movie effect model

#choosing tuning parameter lambda
lambdas <- seq(0, 10, 0.25)

rmses_1 <- sapply(lambdas, function(l){
  
  mu <- mean(train_set$rating)
  
  b_i <- train_set %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  predicted_ratings <- test_set %>% 
    left_join(b_i, by='movieId') %>% 
    mutate(pred = mu + b_i) %>%
    pull(pred)
  return(RMSE(predicted_ratings, test_set$rating))
})
qplot(lambdas, rmses_1)  
lambdas[which.min(rmses_1)] #lambda 2.25

rmse_results <- bind_rows(rmse_results,
                          tibble(method = "Regularised Movie Effect Model",
                                 RMSE = min(rmses_1)))

print.data.frame(rmse_results)

#we can then see that reg movie effects has worked - showing top/worst films
#new top 10 films
movie_titles <- edx %>% 
  select(movieId, title) %>%
  distinct()

train_set %>%
  count(movieId) %>% 
  left_join(movie_reg_avgs, by = "movieId") %>%
  left_join(movie_titles, by = "movieId") %>%
  arrange(desc(b_i)) %>% 
  slice(1:10) %>% 
  pull(title)
#new worst 10 films
train_set %>%
  count(movieId) %>% 
  left_join(movie_reg_avgs, by = "movieId") %>%
  left_join(movie_titles, by="movieId") %>%
  arrange(b_i) %>% 
  select(title, b_i, n) %>% 
  slice(1:10) %>% 
  pull(title)

#model 6 - reg user effect (and movie) - using cross-validation to pick lambda
lambdas <- seq(0, 10, 0.25)
rmses_2 <- sapply(lambdas, function(l){
  
  mu <- mean(train_set$rating)
  
  b_i <- train_set %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- train_set %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  predicted_ratings <- 
    test_set %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  
  return(RMSE(predicted_ratings, test_set$rating))
})

qplot(lambdas, rmses_2)
lambda <- lambdas[which.min(rmses_2)]
lambda
rmse_results <- bind_rows(rmse_results,
                          tibble(method = "Regularised Movie & User Effect Model",
                                 RMSE = min(rmses_2)))

rmse_results %>% knitr::kable()
print.data.frame(rmse_results)
#model 7 - reg movie + user + genres effect
lambdas <- seq(0, 10, 0.25)
rmses_3 <- sapply(lambdas, function(l){
  
  mu <- mean(train_set$rating)
  
  b_i <- train_set %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- train_set %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  b_g <- train_set %>%
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    group_by(genres) %>% 
    summarize(b_g = sum(rating - b_i - b_u - mu)/(n()+l))
  
  predicted_ratings <- test_set %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_g, by= "genres") %>%
    mutate(pred = mu + b_i + b_u + b_g) %>%
    pull(pred)
  
  return(RMSE(predicted_ratings, test_set$rating))
})

qplot(lambdas, rmses_3)
min(rmses_3)
lambda <- lambdas[which.min(rmses_3)] # lambda of 4.75
lambda
rmse_results <- bind_rows(rmse_results,
                          tibble(method = "Regularised Movie & User & Genres Effect Model",
                                 RMSE = min(rmses_3)))

print.data.frame(rmse_results)


##### FINAL MODEL ####
l <- 4.75
mu <- mean(edx$rating)

b_i <- edx %>% 
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+l))

b_u <- edx %>% 
  left_join(b_i, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+l))

b_g <- edx %>%
  left_join(b_i, by="movieId") %>%
  left_join(b_u, by="userId") %>%
  group_by(genres) %>% 
  summarize(b_g = sum(rating - b_i - b_u - mu)/(n()+l))

predicted_ratings <- validation %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_g, by= "genres") %>%
  mutate(pred = mu + b_i + b_u + b_g) %>%
  pull(pred)

rmses_f <- RMSE(predicted_ratings, validation$rating)


qplot(lambdas, rmses_f)
min(rmses_f)
rmse_results <- bind_rows(rmse_results,
                          tibble(method = "Regularised Movie & User & Genres - FINAL",
                                 RMSE = rmses_f))

print.data.frame(rmse_results)
