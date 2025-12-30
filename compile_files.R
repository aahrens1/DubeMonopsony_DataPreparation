library(readr)
library(tidyverse)

dta <- read_csv('../cleaned_data/ipeirotis_cleaned.csv')

dta |> select(reward,duration,title,description) |> 
  filter(!is.na(reward)) |>
  filter(!is.na(duration)) |>
  filter(reward>0 & duration>0) |>
  na.omit()  
