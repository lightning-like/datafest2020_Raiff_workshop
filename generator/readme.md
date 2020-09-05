for group generator you can use GroupGenerator class

GroupGenerator provide a-a test between start and end.
Futhermore GroupGenerator do not use cnums from company banlist and 
narrative_filter.


```
from fxpi_ml.ab_test.population import GroupGenerator
start='2019-01-01'
end='2019-09-29'
narrative_filter = ('CHURN_EXP','NO_DEALS')
groups_size = 180_000
gen = GroupGenerator(
    available_companies=(),
    narrative=narrative_filter,
    )
_data = gen.gen_with_cross_validation(start,
                                      end,
                                      groups=(groups_size,
                                              groups_size,
                                              )
                                     )
```

GroupGenerator retrun best group and remove generated group from general data.
It means you call `gen.gen_with_cross_validation` and create other few groups to
test without intersections

