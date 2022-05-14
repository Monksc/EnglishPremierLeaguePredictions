#!/usr/bin/env sh

./download.bs

python3 format_predictions_to_json_forwebsite.py

((echo -n "const epl2020 = "; cat epl-predictions-stats.json); (cat $HOME/Projects/portfolio/webapp/shared_src/model/EPLPredictionData.js | tail -n+2)) > $HOME/Projects/portfolio/webapp/shared_src/model/EPLPredictionData.js

cd ~/Projects/portfolio/webapp/client && npm run build
cd ~/Projects/portfolio/webapp/ && heroku login && heroku login:container && docker build -t registry.heroku.com/cammonks-portfolio/web . && docker push registry.heroku.com/cammonks-portfolio/web && heroku container:release web -a cammonks-portfolio && heroku open -a cammonks-portfolio


