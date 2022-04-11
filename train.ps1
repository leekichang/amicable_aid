#$dataset = @('motion','keti','wifi','seizure', 'PAMAP2')
$dataset = @('wifi')
$model = @('ResNet', 'MaDNN', 'MaCNN', 'LaxCat', 'RFNet')
foreach ( $d in $dataset )
{
    foreach ($m in $model){
        echo $d
        python ./train.py --dataset $d --model $m
    }
}