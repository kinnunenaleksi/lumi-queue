# Input 

Functionality to preprocess and add features to raw Slurm data.

## Structure

| File    | Description |
| -------- | ------- |
| [`preprocess.py`](preprocess.py)    | Cleans and parses data.  |
| [`features.py`](features.py) | Adds features to the preprocessed data.     |
| [`params.py`](params.py) | Parameters for preprocessing.     |
| [`input.py`](input.py)  | Combines above and returns final datasets.      |

## Usage

The main function of this module is the `input.create_datasets`, that creates the new datasets and
saves them under `data/`. Thus, the following script...

```python
from input.input import create_datasets

written_paths = create_datasets(
    partitions = ['small', 'small-g'],
    truncate_pct = 0.01
)
```
... creates the following folder structure: 

```bash
.
├── data
│   ├── small-g.parquet
│   └── standard.parquet
```
## Raw Input Data 

The input for the predictions comes directly from the Slurm utility `sacct`. The data
preliminarily comes in the following format: 

| Column      | Type    | Description |
| ----------- | ------- | ----------- |
| Account     | Integer | Slurm Account Id. |
| UID   | Integer | Slurm User ID. |
| ReservationID   | Integer | Possible reservation ID for the queue. |
| JobIDRaw   | Integer | Job ID. |
| Priority   | Integer | Priority of given job in the partition. |
| Partition   | String | Partition name, e.g. standard, small, small-g etc. |
| TimelimitRaw   | Integer | Timeout limit for job in minutes. |
| State   | String | Job's final state, i.e. 'Completed', 'Failed' etc. |
| Submit   | String (Timestamp) | Time when job was submitted to Slurm. |
| Eligible   | String (Timestamp) | Time when job could start, i.e. dependencies are loaded. |
| Start   | String (Timestamp) | Start time of the job. |
| End   | String (Timestamp) | End time of the job. |
| ElapsedRaw   | Integer | Job duration in seconds. |
| AllocTRES   | List | List of all resources that were allocated to the job. |

## Preprocessed Input Data 

### Cleaned Input Data

After running `preprocess.py`, the data is transformed in following form: 

| Raw Column | Column | Type | Description | Transformation | Nulls Allowed | Tests |
| ---------- | ------ | ---- | ----------- |--------------- |-------------- | ----- |
| Account | account_id | Integer | Slurm Account Id. | None | False | Values >= 0 |
| UID | user_id   | Integer | Slurm User ID.  | - | False | Values >= 0 |
| ReservationID | reservation_id   | Integer | Possible reservation ID for the queue. | None | True  | None |
| JobIDRaw | job_id   | Integer | Job ID.  | None  | False| Unique |
| Priority | priority | Integer | Priority of given job in the partition. | None | False | Values >= 0 |
| Partition | partition   | String | Partition name, e.g. standard, small, small-g etc. | None | False | Value in known partitions. |
| TimelimitRaw | timelimit_minutes   | Integer | Timeout limit for job in minutes. | None | True | None |
| State | state | String | Job's final state, i.e. 'Completed', 'Failed' etc. | None | False | Filtered to only valid states. |
| Submit | submit_ts   | Timestamp | Time when job was submitted to Slurm. | None | False | submit_ts <= eligible_start_ts |
| Eligible | eligible_start_ts   | Timestamp | Time when job could start, i.e. when dependencies are loaded. | None | False | eligible_start_ts <= start_ts | 
| Start | start_ts  | Timestamp | Start time of the job. | None | False | start_ts <= end_ts |
| End | end_ts   | Timestamp | End time of the job. | None | False | None |
| ElapsedRaw | elapsed_seconds   | Integer | Job duration in seconds. | None | False | Values >= 0 |
| AllocTRES | allocated_cpu   | Integer | Allocated CPUs. | Scraped from `AllocTRES`. | False | Values >= 0 |
| AllocTRES | allocated_node   | Integer | Allocated nodes. | Scraped from `AllocTRES`. | False | Values >= 0 |
| AllocTRES | allocated_gpu   | Integer | Allocated GPU. | Scraped from `AllocTRES`. | True | None |
| AllocTRES | allocated_mem   | Integer | Allocated memory. | Scraped from `AllocTRES`. | True | None |
| AllocTRES | consumed_billing   | Integer | Allocated resources. | Scraped from `AllocTRES`. | False | Values >= 0 |
| AllocTRES | consumed_energy   | Integer | Allocated resources. | Scraped from `AllocTRES`. | False | Values >= 0 |
| Eligible, Start | wait_time_seconds   | Integer | Target variable. | None | False | Values >= 0 |

### Added Features

For the cleaned input, the following features are added in `features.py`: 

| Column    | Type  | Description |
|----------- | ------- | ------------ |
| wait_time_seconds   | Integer | The target variable for analysis. Denotes the time taken between `eligible_start_ts` and `start_ts`. |
| reservation_flag   | Integer | Binary flag for whether job had a reservation or not. |
| chained_flag   | Integer | Binary flag for whether job was chained or not. |
| queued_RESOURCE   | Integer | Total sum of given resource for the jobs in queue before a given job. |
| active_RESOURCE   | Integer | Total sum of given resource for the active jobs during a given job. |
| queued_count_SIZE_jobs*  | Integer | Total count of specific size of jobs in queue before a given job.  |
| active_count_SIZE_jobs*   | Integer |Total count of specific size of jobs in queue before a given job. |

*The size of a given job is determined by either the `allocated_node` or `allocated_cpu` column,
depending on whether the partition is resource or node allocatable. For node allocatable partitions,
the `allocated_node` is used and for resource allocatable partitions, the `allocated_cpu` is used.
The size can be either `small`, `medium` or `large`, which are calculated by <50 percentile, 50-80
percentile, or > 80 percentile for each partition from the determined column.


#### Queued jobs: 

For a given job, jobs in queue are ones submitted before the job, started before
the job, and have ended before the given job has started. Furthermore, they have
priority over the given job. Namely:

- submit_ts < new_submit_ts
- start_ts < new_start_ts
- end_ts > new_start_ts
- priority < current_priority

#### Active jobs: 

Active jobs are ones submitted and started before the new job, and are running
in parallel with the new job.

- submit_ts < new_submit_ts
- start_ts <= new_start_ts
- end_ts < new_start_ts




