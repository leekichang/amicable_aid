#$dataset = @('motion','keti','wifi','seizure', 'PAMAP2')
$dataset = @('keti')
foreach ( $d in $dataset )
{

    echo $d
    python ./train.py --dataset $d --model ResNet
}