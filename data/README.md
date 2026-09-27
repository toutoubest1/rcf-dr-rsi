# External data

No raw data are redistributed in this repository. Download them as follows.

## ASlib 

The ASlib scenarios are the public `coseal/aslib_data` repository (Bischl et al., 2016, *Artificial Intelligence*
237). The analysis used commit `551b22beef8df17de59286b4822ef720e0aa4d6f` (2025-09-29):

```
git clone https://github.com/coseal/aslib_data.git data/aslib/aslib_data
git -C data/aslib/aslib_data checkout 551b22beef8df17de59286b4822ef720e0aa4d6f
```

Scenarios used: `SAT03-16_INDU` (primary) and `ASP-POTASSCO` (replication). The audit
(`real_data/aslib/audit_aslib.py`) reads all runtime scenarios of the clone. About 250 MB.

## KuaiRec 

KuaiRec 2.0 (Gao et al., 2022, CIKM; licence CC BY-SA 4.0) is distributed on Zenodo, record 18164998:

```
mkdir -p data/kuairec
wget -O data/kuairec/KuaiRec.zip https://zenodo.org/records/18164998/files/KuaiRec.zip
#   431,964,858 bytes, md5 261550d472c48eff4990fb13c0e5bcf7
python -m real_data.kuairec.reduce_kuairec data/kuairec/KuaiRec.zip
```

`reduce_kuairec.py` writes `data/kuairec/reduced/` (about 100 MB): the full small matrix, the big-matrix rows of
the 1,411 small-matrix users, `item_categories.csv` and `user_features.csv`. All KuaiRec scripts read only these
files. If `wget` is blocked, download the zip in a browser from https://zenodo.org/records/18164998.
