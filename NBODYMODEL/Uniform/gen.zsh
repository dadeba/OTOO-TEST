#!/bin/zsh

foreach xx (2 4 8 10 15 20 25 30 35 40 45 50 100 200 300 400 500 600 700 800 900 1000 2000 4000 8000)
  echo $xx
  ./run $xx
end