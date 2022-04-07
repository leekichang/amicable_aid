$dataset = @('motion','keti','wifi','seizure')

foreach ( $d in $dataset )
{
    echo $d
    python ./train.py --dataset $d
}