#!/usr/bin/env python3
"""
MLflow Cleanup & Migration Utilities.

Provides utilities to clean up and migrate MLflow runs, including:
- List runs in an experiment with their file paths
- Delete run directories and prepared data
- Migrate runs to an archive location with centralized prep data storage

Usage:
    # List runs
    python mlflow_cleanup.py list --tracking-uri http://127.0.0.1:5001 --experiment "my-experiment"
    
    # Cleanup a run (dry run)
    python mlflow_cleanup.py run --tracking-uri http://127.0.0.1:5001 --run-id abc123 --dry-run
    
    # Migrate a run
    python mlflow_cleanup.py migrate --tracking-uri http://127.0.0.1:5001 --run-id abc123 --destination /path/to/archive
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def get_mlflow_client(tracking_uri: str):
    """Create MLflow client with tracking URI."""
    import mlflow
    from mlflow.tracking import MlflowClient
    
    mlflow.set_tracking_uri(tracking_uri)
    return MlflowClient(tracking_uri=tracking_uri)


def get_run_paths(client, run_id: str) -> Dict[str, Optional[str]]:
    """Fetch run paths from MLflow run params."""
    run = client.get_run(run_id)
    params = run.data.params
    
    return {
        'run_id': run_id,
        'run_name': run.info.run_name,
        'experiment_id': run.info.experiment_id,
        'run_dir': params.get('run_dir'),
        'prepared_data_dir': params.get('prepared_data_dir'),
        'status': run.info.status,
    }


def find_run_by_name(client, run_name: str) -> Optional[Dict[str, Any]]:
    """Search for a run by name across all experiments."""
    experiments = client.search_experiments()
    
    for exp in experiments:
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string=f"run_name = '{run_name}'"
        )
        if runs:
            run = runs[0]
            params = run.data.params
            return {
                'run_id': run.info.run_id,
                'run_name': run.info.run_name,
                'experiment_name': exp.name,
                'status': run.info.status,
                'run_dir': params.get('run_dir'),
                'prepared_data_dir': params.get('prepared_data_dir'),
            }
    
    return None


def resolve_run_id(client, run_id: Optional[str], run_name: Optional[str]) -> Optional[str]:
    """Resolve run ID from either run_id or run_name."""
    if run_id:
        return run_id
    
    if run_name:
        run_info = find_run_by_name(client, run_name)
        if run_info:
            logger.info(f"Found run '{run_name}' in experiment '{run_info['experiment_name']}'")
            return run_info['run_id']
        else:
            logger.error(f"Run not found: {run_name}")
            return None
    
    logger.error("Either --run-id or --run-name must be provided")
    return None


def list_experiment_runs(client, experiment_name: str) -> List[Dict[str, Any]]:
    """List all runs in an experiment with their paths."""
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        logger.error(f"Experiment '{experiment_name}' not found")
        return []
    
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"]
    )
    
    result = []
    for run in runs:
        params = run.data.params
        result.append({
            'run_id': run.info.run_id,
            'run_name': run.info.run_name,
            'status': run.info.status,
            'run_dir': params.get('run_dir'),
            'prepared_data_dir': params.get('prepared_data_dir'),
        })
    
    return result


def delete_directory(path: str, dry_run: bool = True) -> bool:
    """Delete a directory with dry-run support."""
    p = Path(path)
    
    if not p.exists():
        logger.warning(f"Path does not exist (already deleted?): {path}")
        return False
    
    if dry_run:
        logger.info(f"[DRY RUN] Would delete: {path}")
        return True
    
    try:
        shutil.rmtree(p)
        logger.info(f"Deleted: {path}")
        return True
    except Exception as e:
        logger.error(f"Failed to delete {path}: {e}")
        return False


def cleanup_run(
    client,
    run_id: str,
    dry_run: bool = True,
    keep_prep_data: bool = False,
    delete_mlflow_run: bool = False
) -> None:
    """Clean up a single run's files."""
    import mlflow
    
    paths = get_run_paths(client, run_id)
    
    logger.info(f"{'[DRY RUN] ' if dry_run else ''}Cleaning up run: {paths['run_name']} ({run_id})")
    
    # Delete run directory
    if paths['run_dir']:
        delete_directory(paths['run_dir'], dry_run)
    else:
        logger.warning("No run_dir found in run params")
    
    # Delete prepared data (unless --keep-prep-data)
    if not keep_prep_data and paths['prepared_data_dir']:
        delete_directory(paths['prepared_data_dir'], dry_run)
    elif keep_prep_data and paths['prepared_data_dir']:
        logger.info(f"Keeping prepared data: {paths['prepared_data_dir']}")
    
    # Delete MLflow run record
    if delete_mlflow_run:
        if dry_run:
            logger.info(f"[DRY RUN] Would delete MLflow run record: {run_id}")
        else:
            mlflow.set_tracking_uri(client.tracking_uri)
            mlflow.delete_run(run_id)
            logger.info(f"Deleted MLflow run record: {run_id}")


def get_mlflow_artifact_path(client, run_id: str) -> Optional[str]:
    """Get the MLflow artifact storage path for a run."""
    try:
        run = client.get_run(run_id)
        artifact_uri = run.info.artifact_uri
        # Convert file:// URI to path
        if artifact_uri.startswith('file://'):
            return artifact_uri[7:]  # Remove 'file://' prefix
        return artifact_uri
    except Exception as e:
        logger.warning(f"Could not get artifact URI for run {run_id}: {e}")
        return None


def cleanup_run_full(
    client,
    run_id: str,
    dry_run: bool = True,
    keep_prep_data: bool = False,
    delete_mlflow_artifacts: bool = True,
    permanently_delete: bool = False
) -> None:
    """Full cleanup including MLflow artifacts."""
    import mlflow
    
    paths = get_run_paths(client, run_id)
    
    logger.info(f"{'[DRY RUN] ' if dry_run else ''}Full cleanup: {paths['run_name']} ({run_id})")
    
    # Delete run directory
    if paths['run_dir']:
        delete_directory(paths['run_dir'], dry_run)
    
    # Delete prepared data (unless --keep-prep-data)
    if not keep_prep_data and paths['prepared_data_dir']:
        delete_directory(paths['prepared_data_dir'], dry_run)
    
    # Delete MLflow artifacts
    if delete_mlflow_artifacts:
        artifact_path = get_mlflow_artifact_path(client, run_id)
        if artifact_path:
            # Delete the run's artifact directory (go up one level from /artifacts)
            run_artifact_dir = str(Path(artifact_path).parent)
            delete_directory(run_artifact_dir, dry_run)
    
    # Permanently delete the run record
    if permanently_delete:
        if dry_run:
            logger.info(f"[DRY RUN] Would permanently delete run record: {run_id}")
        else:
            # Note: delete_run just marks as deleted, we need to track this
            mlflow.set_tracking_uri(client.tracking_uri)
            mlflow.delete_run(run_id)
            logger.info(f"Marked run as deleted: {run_id}")


def cleanup_experiment(
    client,
    experiment_name: str,
    dry_run: bool = True,
    keep_prep_data: bool = False,
    delete_mlflow_run: bool = False
) -> None:
    """Clean up all runs in an experiment."""
    runs = list_experiment_runs(client, experiment_name)
    
    if not runs:
        logger.info(f"No runs found in experiment: {experiment_name}")
        return
    
    logger.info(f"Found {len(runs)} runs in experiment '{experiment_name}'")
    
    for run_info in runs:
        cleanup_run(
            client,
            run_info['run_id'],
            dry_run=dry_run,
            keep_prep_data=keep_prep_data,
            delete_mlflow_run=delete_mlflow_run
        )


def load_prep_mapping(destination: Path) -> Dict[str, Any]:
    """Load or create prep_mapping.json."""
    mapping_file = destination / "prep_mapping.json"
    if mapping_file.exists():
        with open(mapping_file, 'r') as f:
            return json.load(f)
    return {}


def save_prep_mapping(destination: Path, mapping: Dict[str, Any]) -> None:
    """Save prep_mapping.json."""
    mapping_file = destination / "prep_mapping.json"
    with open(mapping_file, 'w') as f:
        json.dump(mapping, f, indent=2)


def migrate_run(
    client,
    run_id: str,
    destination: str,
    dry_run: bool = True
) -> None:
    """Migrate a run to the destination with centralized prep data."""
    dest = Path(destination)
    paths = get_run_paths(client, run_id)
    
    run_name = paths.get('run_name') or run_id
    run_dir = paths.get('run_dir')
    prep_dir = paths.get('prepared_data_dir')
    
    logger.info(f"{'[DRY RUN] ' if dry_run else ''}Migrating run: {run_name}")
    
    if not run_dir or not Path(run_dir).exists():
        logger.error(f"Run directory not found: {run_dir}")
        return
    
    # Destination paths
    runs_dest = dest / "runs" / run_name
    
    if dry_run:
        logger.info(f"[DRY RUN] Would copy run: {run_dir} -> {runs_dest}")
    else:
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "runs").mkdir(exist_ok=True)
        (dest / "prepared_data").mkdir(exist_ok=True)
        
        # Copy run directory
        if runs_dest.exists():
            logger.warning(f"Destination already exists, skipping: {runs_dest}")
        else:
            shutil.copytree(run_dir, runs_dest)
            logger.info(f"Copied run: {run_dir} -> {runs_dest}")
    
    # Handle prepared data with centralized storage
    if prep_dir and Path(prep_dir).exists():
        prep_name = Path(prep_dir).name
        prep_dest = dest / "prepared_data" / prep_name
        
        # Load mapping
        mapping = load_prep_mapping(dest) if not dry_run else {}
        
        if prep_name in mapping:
            # Prep data already migrated, just update mapping
            if dry_run:
                logger.info(f"[DRY RUN] Prep data already exists, would add run to mapping: {prep_name}")
            else:
                if run_name not in mapping[prep_name]['used_by_runs']:
                    mapping[prep_name]['used_by_runs'].append(run_name)
                    save_prep_mapping(dest, mapping)
                logger.info(f"Prep data already exists, added run to mapping: {prep_name}")
        else:
            # Copy prep data and create mapping entry
            if dry_run:
                logger.info(f"[DRY RUN] Would copy prep data: {prep_dir} -> {prep_dest}")
            else:
                shutil.copytree(prep_dir, prep_dest)
                mapping[prep_name] = {
                    'source_path': prep_dir,
                    'migrated_path': str(prep_dest),
                    'used_by_runs': [run_name]
                }
                save_prep_mapping(dest, mapping)
                logger.info(f"Copied prep data: {prep_dir} -> {prep_dest}")
        
        # Update paths.json in migrated run
        if not dry_run:
            new_paths_file = runs_dest / "paths.json"
            if new_paths_file.exists():
                with open(new_paths_file, 'r') as f:
                    paths_data = json.load(f)
                paths_data['prepared_data_dir'] = str(prep_dest)
                paths_data['_original_prepared_data_dir'] = prep_dir
                with open(new_paths_file, 'w') as f:
                    json.dump(paths_data, f, indent=2)
                logger.info(f"Updated paths.json with new prep data location")


def migrate_experiment(
    client,
    experiment_name: str,
    destination: str,
    dry_run: bool = True
) -> None:
    """Migrate all runs in an experiment."""
    runs = list_experiment_runs(client, experiment_name)
    
    if not runs:
        logger.info(f"No runs found in experiment: {experiment_name}")
        return
    
    logger.info(f"Found {len(runs)} runs in experiment '{experiment_name}'")
    
    for run_info in runs:
        migrate_run(client, run_info['run_id'], destination, dry_run)


def cmd_list(args) -> None:
    """Handle 'list' command."""
    client = get_mlflow_client(args.tracking_uri)
    runs = list_experiment_runs(client, args.experiment)
    
    if not runs:
        print(f"No runs found in experiment: {args.experiment}")
        return
    
    print(f"\n{'='*80}")
    print(f"Experiment: {args.experiment} ({len(runs)} runs)")
    print(f"{'='*80}\n")
    
    for run in runs:
        print(f"Run: {run['run_name']}")
        print(f"  ID: {run['run_id']}")
        print(f"  Status: {run['status']}")
        print(f"  Run Dir: {run['run_dir'] or 'N/A'}")
        print(f"  Prep Dir: {run['prepared_data_dir'] or 'N/A'}")
        print()


def cmd_run(args) -> None:
    """Handle 'run' cleanup command."""
    client = get_mlflow_client(args.tracking_uri)
    run_id = resolve_run_id(client, getattr(args, 'run_id', None), getattr(args, 'run_name', None))
    if not run_id:
        return
    cleanup_run(
        client,
        run_id,
        dry_run=args.dry_run,
        keep_prep_data=args.keep_prep_data,
        delete_mlflow_run=args.delete_mlflow_run
    )


def cmd_experiment(args) -> None:
    """Handle 'experiment' cleanup command."""
    client = get_mlflow_client(args.tracking_uri)
    cleanup_experiment(
        client,
        args.experiment,
        dry_run=args.dry_run,
        keep_prep_data=args.keep_prep_data,
        delete_mlflow_run=args.delete_mlflow_run
    )


def cmd_migrate(args) -> None:
    """Handle 'migrate' command."""
    client = get_mlflow_client(args.tracking_uri)
    run_id = resolve_run_id(client, getattr(args, 'run_id', None), getattr(args, 'run_name', None))
    if not run_id:
        return
    migrate_run(client, run_id, args.destination, dry_run=args.dry_run)


def cmd_migrate_experiment(args) -> None:
    """Handle 'migrate-experiment' command."""
    client = get_mlflow_client(args.tracking_uri)
    migrate_experiment(client, args.experiment, args.destination, dry_run=args.dry_run)


def find_deleted_runs(client) -> List[Dict[str, Any]]:
    """Find all runs that need cleanup: deleted runs in active experiments + all runs in deleted experiments."""
    from mlflow.entities import ViewType
    
    deleted_runs = []
    
    # 1. Find deleted runs in ACTIVE experiments
    active_experiments = client.search_experiments(view_type=ViewType.ACTIVE_ONLY)
    for exp in active_experiments:
        try:
            runs = client.search_runs(
                experiment_ids=[exp.experiment_id],
                run_view_type=ViewType.DELETED_ONLY
            )
            for run in runs:
                params = run.data.params
                deleted_runs.append({
                    'run_id': run.info.run_id,
                    'run_name': run.info.run_name,
                    'experiment_name': exp.name,
                    'status': run.info.status,
                    'run_dir': params.get('run_dir'),
                    'prepared_data_dir': params.get('prepared_data_dir'),
                })
        except Exception as e:
            logger.warning(f"Error searching experiment {exp.name}: {e}")
    
    # 2. Find ALL runs in DELETED experiments (they're all orphaned)
    deleted_experiments = client.search_experiments(view_type=ViewType.DELETED_ONLY)
    for exp in deleted_experiments:
        try:
            runs = client.search_runs(
                experiment_ids=[exp.experiment_id],
                run_view_type=ViewType.ALL  # Get all runs in deleted experiment
            )
            for run in runs:
                params = run.data.params
                deleted_runs.append({
                    'run_id': run.info.run_id,
                    'run_name': run.info.run_name,
                    'experiment_name': f"{exp.name} [DELETED EXP]",
                    'status': run.info.status,
                    'run_dir': params.get('run_dir'),
                    'prepared_data_dir': params.get('prepared_data_dir'),
                })
        except Exception as e:
            logger.warning(f"Error searching deleted experiment {exp.name}: {e}")
    return deleted_runs


def garbage_collect(
    client,
    dry_run: bool = True,
    keep_prep_data: bool = False,
    permanently_delete: bool = False
) -> None:
    """Find and clean up all deleted runs."""
    deleted_runs = find_deleted_runs(client)
    
    if not deleted_runs:
        logger.info("No deleted runs found. Nothing to clean up.")
        return
    
    logger.info(f"Found {len(deleted_runs)} deleted runs to clean up")
    
    for run_info in deleted_runs:
        logger.info(f"  - {run_info['run_name']} (exp: {run_info['experiment_name']})")
    
    print()  # Blank line for readability
    
    for run_info in deleted_runs:
        cleanup_run_full(
            client,
            run_info['run_id'],
            dry_run=dry_run,
            keep_prep_data=keep_prep_data,
            delete_mlflow_artifacts=True,
            permanently_delete=permanently_delete
        )


def cmd_gc(args) -> None:
    """Handle 'gc' (garbage collection) command."""
    client = get_mlflow_client(args.tracking_uri)
    garbage_collect(
        client,
        dry_run=args.dry_run,
        keep_prep_data=args.keep_prep_data,
        permanently_delete=args.permanently_delete
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MLflow Cleanup & Migration Utilities",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Common arguments
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--tracking-uri', required=True, help='MLflow tracking server URI')
    
    # 'list' command
    p_list = subparsers.add_parser('list', parents=[common], help='List runs in an experiment')
    p_list.add_argument('--experiment', required=True, help='Experiment name')
    p_list.set_defaults(func=cmd_list)
    
    # 'run' cleanup command
    p_run = subparsers.add_parser('run', parents=[common], help='Cleanup a single run')
    run_id_group = p_run.add_mutually_exclusive_group(required=True)
    run_id_group.add_argument('--run-id', help='MLflow run ID')
    run_id_group.add_argument('--run-name', help='Run name (folder name, searches all experiments)')
    p_run.add_argument('--dry-run', action='store_true', default=True,
                       help='Show what would be deleted without deleting (default: True)')
    p_run.add_argument('--no-dry-run', dest='dry_run', action='store_false',
                       help='Actually delete files')
    p_run.add_argument('--keep-prep-data', action='store_true',
                       help='Keep prepared data directory')
    p_run.add_argument('--delete-mlflow-run', action='store_true',
                       help='Also delete the run record from MLflow tracking DB')
    p_run.set_defaults(func=cmd_run)
    
    # 'experiment' cleanup command
    p_exp = subparsers.add_parser('experiment', parents=[common], help='Cleanup all runs in an experiment')
    p_exp.add_argument('--experiment', required=True, help='Experiment name')
    p_exp.add_argument('--dry-run', action='store_true', default=True,
                       help='Show what would be deleted without deleting (default: True)')
    p_exp.add_argument('--no-dry-run', dest='dry_run', action='store_false',
                       help='Actually delete files')
    p_exp.add_argument('--keep-prep-data', action='store_true',
                       help='Keep prepared data directories')
    p_exp.add_argument('--delete-mlflow-run', action='store_true',
                       help='Also delete run records from MLflow tracking DB')
    p_exp.set_defaults(func=cmd_experiment)
    
    # 'migrate' command
    p_migrate = subparsers.add_parser('migrate', parents=[common], help='Migrate a run to archive')
    migrate_id_group = p_migrate.add_mutually_exclusive_group(required=True)
    migrate_id_group.add_argument('--run-id', help='MLflow run ID')
    migrate_id_group.add_argument('--run-name', help='Run name (folder name, searches all experiments)')
    p_migrate.add_argument('--destination', required=True, help='Destination directory')
    p_migrate.add_argument('--dry-run', action='store_true', default=True,
                           help='Show what would be migrated without migrating (default: True)')
    p_migrate.add_argument('--no-dry-run', dest='dry_run', action='store_false',
                           help='Actually migrate files')
    p_migrate.set_defaults(func=cmd_migrate)
    
    # 'migrate-experiment' command
    p_migrate_exp = subparsers.add_parser('migrate-experiment', parents=[common],
                                          help='Migrate all runs in an experiment')
    p_migrate_exp.add_argument('--experiment', required=True, help='Experiment name')
    p_migrate_exp.add_argument('--destination', required=True, help='Destination directory')
    p_migrate_exp.add_argument('--dry-run', action='store_true', default=True,
                               help='Show what would be migrated without migrating (default: True)')
    p_migrate_exp.add_argument('--no-dry-run', dest='dry_run', action='store_false',
                               help='Actually migrate files')
    p_migrate_exp.set_defaults(func=cmd_migrate_experiment)
    
    # 'gc' (garbage collection) command
    p_gc = subparsers.add_parser('gc', parents=[common],
                                 help='Clean up deleted runs (garbage collection)')
    p_gc.add_argument('--dry-run', action='store_true', default=True,
                      help='Show what would be cleaned without cleaning (default: True)')
    p_gc.add_argument('--no-dry-run', dest='dry_run', action='store_false',
                      help='Actually clean up files')
    p_gc.add_argument('--keep-prep-data', action='store_true',
                      help='Keep prepared data directories')
    p_gc.add_argument('--permanently-delete', action='store_true',
                      help='Permanently delete run records from MLflow (not just files)')
    p_gc.set_defaults(func=cmd_gc)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)


if __name__ == "__main__":
    main()
