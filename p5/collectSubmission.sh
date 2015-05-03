rm -f pa5.zip
zip -r pa5.zip . -x "*.git*" "*cs4670/datasets*" "*.ipynb_checkpoints*" "*README.md" "*collectSubmission.sh" "*requirements.txt" ".env/*" "*.pyc" "*cs4670/build/*"
