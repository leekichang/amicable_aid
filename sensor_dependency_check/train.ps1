$dataset = (0..6)
$model = @('ResNet', 'MaDNN', 'MaCNN')
#$model = @('MaCNN')

foreach ( $d in $dataset )
{
    foreach ($m in $model){
        echo $d $m
        python ./train.py --dataset $d --model $m
    }
}

./test.ps1