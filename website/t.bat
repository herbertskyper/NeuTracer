@echo off
cd .\docs\.vuepress\dist
git init
git add .
git commit -m "deploy"
git branch -M gh-pages
git remote add origin git@github.com:lightbreezz/NeuTracer_Tutorial.git
git push -f origin gh-pages
pause