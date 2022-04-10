$dataset = @('motion','keti','wifi','seizure', 'PAMAP2')
$model = @('MaCNN', 'MaDNN', 'ResNet')

foreach ( $d in $dataset )
{
    foreach( $m in $model )
    {
        echo $d
        echo $m
        python ./plot.py --dataset $d --model $m
    
    }
}