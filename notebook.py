# Databricks notebook source
import json, hashlib
from datetime import datetime, timezone
from pyspark.sql import functions as F
INPUT = "training.to6_challenge3_inputs"
OUTPUT = "training.to6_native_eda_20261119"
required = ["return_id", "tax_year", "income", "deduction"]
def rows(df):
    return [r.asDict(recursive=True) for r in df.collect()]
def encode(x):
    return json.dumps(x, indent=2, sort_keys=True, default=str)
# Read source metadata and the supplied traceability view, not just its name.
metadata = {name: rows(spark.sql(f"DESCRIBE TABLE EXTENDED {INPUT}.{name}")) for name in ["returns_initial", "returns_corrected"]}
lineage = rows(spark.table(f"{INPUT}.source_lineage"))
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {OUTPUT}")
def profile(df):
    return {
        "row_count": df.count(),
        "years": rows(df.groupBy("tax_year").count().orderBy("tax_year")),
        "missing": df.agg(*[F.sum(F.col(c).isNull().cast("int")).alias(c) for c in required]).first().asDict(),
        "duplicates": rows(df.groupBy("return_id").count().filter("count > 1").orderBy("return_id")),
        "zero_income": rows(df.filter("income = 0").select("return_id", "tax_year").orderBy("return_id")),
    }
def violations(df):
    duplicate_ids = df.groupBy("return_id").count().filter("count > 1").select("return_id").withColumn("duplicate", F.lit(True))
    marked = df.join(duplicate_ids, "return_id", "left")
    checks = [F.when(F.col(c).isNull(), F.lit("missing:" + c)) for c in required]
    checks += [F.when(F.col("duplicate"), F.lit("duplicate:return_id")), F.when(F.col("income") < 0, F.lit("negative:income")), F.when(F.col("deduction") < 0, F.lit("negative:deduction"))]
    return marked.withColumn("reasons", F.filter(F.array(*checks), lambda x: x.isNotNull())).filter(F.size("reasons") > 0).drop("duplicate")
initial = spark.table(f"{INPUT}.returns_initial")
corrected = spark.table(f"{INPUT}.returns_corrected")
eda = {"initial": profile(initial), "corrected": profile(corrected)}
rejected = rows(violations(initial).orderBy("return_id"))
assert rejected, "The initial batch must fail the gate"
# A rejected batch never enters the writer. Retain all bad records instead of deleting them to pass.
assert violations(corrected).count() == 0, "Replacement must pass exactly the same checks"
interpretation = (
    f"Initial full batch: {eda['initial']['row_count']} rows, year counts {eda['initial']['years']}. "
    f"Duplicate IDs {eda['initial']['duplicates']} and missing counts {eda['initial']['missing']} require rejecting the initial batch. "
    f"Zero-income records {eda['initial']['zero_income']} are valid, not missing; preserve them with a null ratio. "
    f"Replacement full batch has {eda['corrected']['row_count']} rows, {eda['corrected']['duplicates']} duplicate groups and missing counts {eda['corrected']['missing']}; apply the identical gate before writing. "
    f"Its observed year counts {eda['corrected']['years']} determine the 2022/2023/2024 split coverage."
)
print(encode(eda)); print(interpretation)
def deliver():
    stamp = datetime.now(timezone.utc).isoformat()
    assert violations(corrected).count() == 0
    features = corrected.select("return_id", "tax_year", F.when(F.col("income") == 0, F.lit(None).cast("double")).otherwise(F.col("deduction") / F.col("income")).alias("deduction_to_income"))
    features.write.mode("overwrite").saveAsTable(f"{OUTPUT}.features")
    saved = spark.table(f"{OUTPUT}.features")
    assert saved.columns == ["return_id", "tax_year", "deduction_to_income"]
    assert saved.count() == corrected.count() == saved.select("return_id").distinct().count()
    # Independent scalar calculation from the source, compared to the saved table.
    example = corrected.filter("income > 0").orderBy("return_id").first()
    expected = float(example.deduction) / float(example.income)
    actual = saved.filter(F.col("return_id") == example.return_id).first().deduction_to_income
    assert abs(actual - expected) < 1e-12
    zero_ids = {r.return_id for r in corrected.filter("income = 0").select("return_id").collect()}
    assert {r.return_id for r in saved.filter("deduction_to_income IS NULL").select("return_id").collect()} == zero_ids
    exports = {"features": rows(saved.orderBy("return_id"))}
    for name, year in [("train", 2022), ("validation", 2023), ("test", 2024)]:
        saved.filter(F.col("tax_year") == year).write.mode("overwrite").saveAsTable(f"{OUTPUT}.{name}")
        exports[name] = rows(spark.table(f"{OUTPUT}.{name}").orderBy("return_id"))
        assert all(r["tax_year"] == year for r in exports[name])
    combined = sorted([r for n in ["train", "validation", "test"] for r in exports[n]], key=lambda r: r["return_id"])
    assert combined == exports["features"]
    return {"started_at": stamp, "finished_at": datetime.now(timezone.utc).isoformat(), "tables": exports, "independent_ratio": {"return_id": example.return_id, "expected": expected, "saved": actual}}
first = deliver()
second = deliver()
assert first["tables"] == second["tables"], "Full IDs and values must match across successful passes"
files = {
    "outputs/eda.json": encode(eda), "outputs/rejected.json": encode(rejected),
    "outputs/source-metadata.json": encode(metadata), "outputs/lineage.json": encode(lineage),
    "outputs/run-1.json": encode(first), "outputs/run-2.json": encode(second),
    "outputs/interpretation.txt": interpretation,
    "SUBMISSION.md": f"""# Native Databricks EDA acceptance submission
Inputs: {INPUT}.returns_initial and {INPUT}.returns_corrected. Supplied traceability: {INPUT}.source_lineage, read into outputs/lineage.json; source descriptions retained in outputs/source-metadata.json. One row is one return, not one taxpayer.

{interpretation}

Initial rejection evidence is outputs/rejected.json, preserving the affected rows and reasons. No initial batch write occurs. The replacement passes the same required-value, unique-ID and nonnegative checks. Valid zeros are retained.

Output tables: {OUTPUT}.features, {OUTPUT}.train, {OUTPUT}.validation, {OUTPUT}.test. Ratio = deduction/income, null for zero income. Outputs contain only return_id, tax_year and deduction_to_income; no taxpayer identifiers or raw amounts. The readback verifies a separately calculated nonzero ratio and zero-income IDs.

Two successful complete pipeline passes within this native job: {first['started_at']} and {second['started_at']}. Full saved IDs and values are retained in outputs/run-1.json and outputs/run-2.json and compared for equality. The union of year tables must exactly equal the feature table; overwrite prevents appended duplicates. The job URL and ID are recorded by the submission transport in outputs/native-run.json.

This is an operator acceptance artifact, not a learner session or student-permission test. It does not establish Ownership or Trait. No model training or chart is required or claimed.
""",
}
dbutils.notebook.exit(json.dumps({"files": files}))
