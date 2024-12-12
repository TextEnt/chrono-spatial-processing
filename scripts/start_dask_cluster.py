from dask.distributed import LocalCluster

def main():
    cluster = LocalCluster()
    print(cluster.scheduler_address)

if __name__ == '__main__':
    main()