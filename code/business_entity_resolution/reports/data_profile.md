# Dataset inspection

All inputs were read as tab-separated files. Counts below cover the full files; duplicate and character statistics use the first 100,000 shuffled records of each source.

| File | Rows | Country mix | Missing address | Mean name length | Mean address length |
|---|---:|---|---:|---:|---:|
| train Source 1 | 2,206,821 | US 1,323,633; India 883,188 | 0% | 24.03 | 52.07 |
| train Source 2 | 5,034,616 | US 3,016,817; India 2,017,799 | 3.356% | 25.10 | 46.23 |
| train Source 3 | 5,285,603 | US 3,170,056; India 2,115,547 | 3.328% | 25.20 | 46.71 |
| test Source 1 | 1,732,544 | India 809,986; US 663,106; France 259,452 | 0% | 23.84 | 57.21 |
| test Source 2 | 4,887,273 | India 2,312,565; US 1,871,330; France 703,378 | 2.648% | 25.70 | 50.41 |
| test Source 3 | 5,082,316 | India 2,405,000; US 1,945,701; France 731,615 | 2.678% | 25.66 | 48.74 |

No file had missing names or countries. The ground truth has one row for each training Source 1 entity: 123,247 empty lists (5.59%), 119,157 single matches (5.40%), and 1,964,417 multi matches (89.01%). It contains 3,693,619 links to Source 2 and 3,944,746 to Source 3. Match counts range from 0 to 11, with 3 and 4 most common.

| True links per Source 1 | Entities |
|---:|---:|
| 0 | 123,247 |
| 1 | 119,157 |
| 2 | 375,212 |
| 3 | 530,841 |
| 4 | 484,115 |
| 5 | 321,957 |
| 6 | 164,868 |
| 7 | 63,968 |
| 8 | 18,680 |
| 9 | 4,205 |
| 10 | 534 |
| 11 | 37 |

In a random sample of 69,439 true links, 25.70% have exactly equal punctuation-normalized names and 8.17% exactly equal normalized addresses. Exact field matching alone is therefore insufficient. Name edit similarity exceeds 70 for 78.9% of sampled links; 6.7% have both name and address edit similarity below 70. Some difficult matches involve a Latin name paired with an Indian-script translation and a near-identical address. Others have a completely different trading name but nearly identical address.

Examples observed in that labeled sample:

- `Mumbai Marketing Private Limited` matches `Onyxcira` despite name similarity near 15/100; the addresses are near 91/100.
- `All Engineering Pvt Ltd` matches `ऑल इंजीनियरिंग प्रा. लि.`; the addresses are near 93/100.
- `Smith Safe Alpex Inc` matches the same normalized name in Source 3 even though the target address is missing.

In 100,000-row samples, duplicate normalized names occur in about 10% of Source 1 rows and 1.8–2.4% of Source 2/3 rows. Duplicate normalized addresses occur in about 0.3% of Source 1 rows and 3–3.7% of Source 2/3 rows. About 90–97% of addresses contain digits, averaging roughly 1.5 numeric tokens. Test Source 1 has 2.32% non-ASCII names, while training Source 1 has almost none; French accents and Indian scripts appear in targets. These observations favor separate name and address retrieval, Unicode-aware tokenization, numeric agreement features, and an open-set country equality feature.
