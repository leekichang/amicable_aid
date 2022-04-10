$dataset = @('keti')
$model = @('ResNet', 'MaDNN', 'MaCNN')
$aid = @('False', 'True')

foreach ( $d in $dataset )
{
    foreach( $m in $model )
    {
        foreach( $a in $aid){
            echo $d
            echo $m
            echo $a
            python ./adversarial_attack.py --dataset $d --model $m --aid $a > .\results\$m\$d'_'$a.txt
        }
    }
}
./plot.ps1