$dataset = (0..6)
$testset = (0..2)
$model = @('ResNet', 'MaDNN', 'MaCNN')
#$model = @('MaCNN')

foreach ( $m in $model )
{   
    foreach ( $d in $dataset ){
        foreach ( $t in $testset ){
            echo $d $t $m
            python ./test.py --dataset $d --model $m --testset $t --norm True
        }
    }
}