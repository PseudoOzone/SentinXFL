"""
SentinXFL Command Line Interface
=================================

Provides CLI commands for running the API server, PII pipeline, and other operations.

Usage:
    python -m sentinxfl.cli server     # Start API server
    python -m sentinxfl.cli scan       # Run PII scan on datasets
    python -m sentinxfl.cli certify    # Run certification pipeline

Author: Anshuman Bakshi
"""

import sys
from pathlib import Path

import click

# Add src to path if needed
src_path = Path(__file__).parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


@click.group()
@click.version_option(version="2.0.0", prog_name="SentinXFL")
def cli():
    """SentinXFL - Privacy-First Federated Fraud Detection Platform."""
    pass


@cli.command()
@click.option("--host", default="0.0.0.0", help="Server host")
@click.option("--port", default=8000, type=int, help="Server port")
@click.option("--reload", is_flag=True, help="Enable auto-reload")
def server(host: str, port: int, reload: bool):
    """Start the API server."""
    import uvicorn

    click.echo(f"Starting SentinXFL server on {host}:{port}")
    uvicorn.run(
        "sentinxfl.api.main:app",
        host=host,
        port=port,
        reload=reload,
    )


@cli.command()
@click.option("--dataset", type=click.Choice(["bank", "creditcard", "paysim", "all"]), default="all")
@click.option("--sample", default=0.01, type=float, help="Sample fraction (0.0-1.0)")
@click.option("--output", default=None, help="Output report path")
def scan(dataset: str, sample: float, output: str):
    """Scan datasets for PII."""
    from sentinxfl.data.loader import DataLoader
    from sentinxfl.data.schemas import DatasetType
    from sentinxfl.privacy.detector import PIIDetector

    click.echo("SentinXFL PII Scanner")
    click.echo("=" * 50)

    detector = PIIDetector(strict_mode=True)
    loader = DataLoader()

    datasets_to_scan = {
        "bank": [DatasetType.BANK_ACCOUNT_FRAUD],
        "creditcard": [DatasetType.CREDIT_CARD_FRAUD],
        "paysim": [DatasetType.PAYSIM],
        "all": [DatasetType.BANK_ACCOUNT_FRAUD, DatasetType.CREDIT_CARD_FRAUD, DatasetType.PAYSIM],
    }

    for dtype in datasets_to_scan[dataset]:
        click.echo(f"\nScanning {dtype.value}...")
        try:
            if dtype == DatasetType.BANK_ACCOUNT_FRAUD:
                df = loader.load_bank_account_fraud(sample_frac=sample)
            elif dtype == DatasetType.CREDIT_CARD_FRAUD:
                df = loader.load_credit_card_fraud(sample_frac=sample)
            elif dtype == DatasetType.PAYSIM:
                df = loader.load_paysim(sample_frac=sample)

            result = detector.detect(df)
            report = detector.generate_report(result)
            click.echo(report)

            if output:
                output_path = Path(output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "w") as f:
                    f.write(report)
                click.echo(f"Report saved to {output_path}")

        except FileNotFoundError as e:
            click.echo(f"Dataset not found: {e}", err=True)
        except Exception as e:
            click.echo(f"Error: {e}", err=True)


@cli.command()
@click.option("--dataset", type=click.Choice(["bank", "creditcard", "paysim"]), required=True)
@click.option("--sample", default=0.1, type=float, help="Sample fraction for testing")
def certify(dataset: str, sample: float):
    """Run full certification pipeline on a dataset."""
    from sentinxfl.data.loader import DataLoader
    from sentinxfl.data.schemas import DatasetType
    from sentinxfl.privacy.detector import PIIDetector
    from sentinxfl.privacy.transformer import PIITransformer
    from sentinxfl.privacy.certifier import PIICertifier
    from sentinxfl.privacy.audit import PIIAuditLog

    click.echo("SentinXFL 5-Gate Certification Pipeline")
    click.echo("=" * 50)

    dataset_map = {
        "bank": DatasetType.BANK_ACCOUNT_FRAUD,
        "creditcard": DatasetType.CREDIT_CARD_FRAUD,
        "paysim": DatasetType.PAYSIM,
    }

    dtype = dataset_map[dataset]

    # Initialize components
    loader = DataLoader()
    detector = PIIDetector(strict_mode=True)
    transformer = PIITransformer()
    certifier = PIICertifier()
    audit = PIIAuditLog(actor="cli")

    try:
        # Step 1: Load
        click.echo(f"\n[1/4] Loading {dtype.value}...")
        if dtype == DatasetType.BANK_ACCOUNT_FRAUD:
            df = loader.load_bank_account_fraud(sample_frac=sample)
        elif dtype == DatasetType.CREDIT_CARD_FRAUD:
            df = loader.load_credit_card_fraud(sample_frac=sample)
        elif dtype == DatasetType.PAYSIM:
            df = loader.load_paysim(sample_frac=sample)

        audit.log_scan_started(dtype.value, len(df.columns), len(df))
        click.echo(f"    Loaded {len(df)} rows, {len(df.columns)} columns")

        # Step 2: Detect
        click.echo("\n[2/4] Running PII detection...")
        detection_result = detector.detect(df)
        audit.log_scan_completed(dtype.value, detection_result)
        click.echo(f"    Found {detection_result.columns_with_pii} columns with PII")
        click.echo(f"    Detection passed: {detection_result.passed}")

        # Step 3: Transform
        click.echo("\n[3/4] Transforming PII...")
        transformed_df, transform_results = transformer.transform(df, detection_result)
        audit.log_transformation_completed(dtype.value, transform_results)
        successful = sum(1 for r in transform_results if r.success)
        click.echo(f"    Transformed {successful}/{len(transform_results)} columns")

        # Step 4: Certify
        click.echo("\n[4/4] Running certification...")
        new_detection = detector.detect(transformed_df)
        cert_result = certifier.certify(transformed_df, new_detection)
        audit.log_certification(dtype.value, cert_result)

        # Print certificate
        certificate = certifier.generate_certificate(cert_result)
        click.echo("\n" + certificate)

        # Export audit
        audit_path = audit.export_json()
        click.echo(f"\nAudit trail exported to: {audit_path}")

    except FileNotFoundError as e:
        click.echo(f"Dataset not found: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
def info():
    """Show system and configuration info."""
    from sentinxfl.core.config import settings
    import platform

    click.echo("SentinXFL System Information")
    click.echo("=" * 50)
    click.echo(f"App Name: {settings.app_name}")
    click.echo(f"Version: {settings.app_version}")
    click.echo(f"Environment: {settings.environment}")
    click.echo(f"Python: {platform.python_version()}")
    click.echo(f"Platform: {platform.system()} {platform.release()}")

    click.echo("\nPaths:")
    click.echo(f"  Data Dir: {settings.data_dir_abs}")
    click.echo(f"  Processed Dir: {settings.processed_dir_abs}")
    click.echo(f"  Models Dir: {settings.models_dir_abs}")

    click.echo("\nPrivacy Settings:")
    click.echo(f"  DP Epsilon: {settings.dp_epsilon}")
    click.echo(f"  DP Delta: {settings.dp_delta}")
    click.echo(f"  PII Strict Mode: {settings.pii_strict_mode}")

    # Check GPU
    try:
        import torch
        if torch.cuda.is_available():
            click.echo("\nGPU:")
            click.echo(f"  Name: {torch.cuda.get_device_name(0)}")
            click.echo(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            click.echo("\nGPU: Not available")
    except ImportError:
        click.echo("\nGPU: PyTorch not installed")


def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
