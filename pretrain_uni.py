from nmt_uni import train
from config import pretrain_config


def main(job_id, params):
    print 'pretraining settings:'
    for c, v in sorted(params.items(), key=lambda a:a[0]):
        print '{}: {}'.format(c, v)

    validerr = train(**params)
    return validerr

if __name__ == '__main__':
    main(0, pretrain_config())


