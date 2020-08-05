# Contributing

This is a guide that lays out how to contribute additional datasets or fixes to this dataset_prep repository.

1. Make sure you have the latest updates from the `master` branch. This is the branch that is known as the ground truth and should never be in a broken state.

```bash
git checkout master
git pull
```

2. Create a feature branch with a descriptive name off of `master`.

```bash
git checkout -b <dataset_new>_pipeline
```

3. Do your work in your feature branch with occassionally pushing up using the `git add`, `git commit`, `git push` flow. When you make your first push to your branch, you should open up a merge request on the repository going from your branch to `master`. As you open this up, prefix your branch with `WIP:`, which stands for "Work In Progress" in order for others to know that you are not ready for a finaly code review yet. Example: `WIP: <dataset_name> Pipeline`. As you push code here, feel free to tag people for thoughts and questions in the MR. This allows dialouge to happen closer to the code and you can give suggestions while tagging specific lines of code. You should have the issue(s) linked to the MR as well.

4. Once your pipeline is ready and you've tested various components of it with either calling the functions directly or using the `proc.py` file at the root of this repo, make sure you have all of the latest changes by doing a rebase from `master` and solve any necessary merge conflicts at this step. You should also be doing this pretty regularly as you develop to make sure you always have the latest code. DO NOT run the `transfer` step until you get a code review and everything looks good for larger datasets as to not make that a blocker for idle waiting time as well as to be cognizant of S3 data transfer costs.

```bash
git checkout master
git pull
git checkout <feature_branch_name>
git rebase master
```

5. Once you feel like you are done with all of the functionality that is going into your branch, you should update the `HISTORY.md` file in the appropriate location for if it is a bug fix, feature addition, etc.

6. Now, remove the `WIP:` prefix from the MR name. and tag either `Mark Hoffmann` or `Alice Yepremyan` in the 'Assigned To'. This signifies that you code is tested to your knowledge, good for a final review, and is ready to be merged into `master`. NOTE: Before doing this step and removing `WIP:`, the CI should be passing. If the CI is not passing and you need help, tag someone in the comments on the MR for help, but don't request a final review.

7. Run a final push of your pipeline to get the dataset fully synced to the infrastructure. The dataset gets synced by default to the `/scratch` directory in the `lwll-datasets` bucket. We do this so that for some of the larger datasets that take a long time to push to the cloud, a performer doesn't try and download that same dataset and end up with corrupt data. You should check that the appropriate dataset metadata is in Firebase and that your compressed tar.gz file along with label files have made their way successfully into the S3 bucket.

8. After your dataset gets synced, ask either `Mark Hoffmann` or `Alice Yepremyan` to sync your data into the `/live` directory within the S3 bucket. (This process will eventually be more automated in the future as well)

9. Your branch will be merged by the code reviewer at this point.

**Merging Details**

When merging, we:

1. Delete source branch on merge
2. DO NOT squash commits
