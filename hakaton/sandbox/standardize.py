import os
from datetime import datetime
from hakaton.preprocessing.standardization import Standardizer

if __name__ == '__main__':

    standardizer = Standardizer()

    scaler = standardizer.fit_transform(
        'std',
        ['/home/mtkaleta/ds320x256']
    )

    now = datetime.now()
    time = now.strftime('%H%M')
    standardizer.save(os.path.join('std', time + 'standardizer.pkl'))
