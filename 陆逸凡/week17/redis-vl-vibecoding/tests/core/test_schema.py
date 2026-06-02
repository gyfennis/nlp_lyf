"""Tests for IndexSchema and IndexManager."""

import pytest

from vecstore.core.schema import (
    DistanceMetric,
    IndexSchema,
    IndexType,
    NumericField,
    TagField,
    TextField,
    VectorField,
)


class TestIndexSchema:
    """Tests for IndexSchema command generation."""

    def test_vector_field_defaults(self):
        """VectorField should have sensible defaults."""
        f = VectorField(name="emb", dimensions=768)
        assert f.name == "emb"
        assert f.dimensions == 768
        assert f.algorithm == IndexType.FLAT
        assert f.distance_metric == DistanceMetric.COSINE

    def test_minimal_schema_ft_create(self):
        """A schema with one vector field should generate valid FT.CREATE args."""
        schema = IndexSchema(
            index_name="test_idx",
            prefix="test:",
            vector_fields=[VectorField(name="vec", dimensions=4)],
        )
        args = schema.build_ft_create_args()
        assert args[0] == "test_idx"
        assert "ON" in args
        assert "HASH" in args
        assert "PREFIX" in args
        assert "1" in args
        assert "test:" in args
        assert "SCHEMA" in args
        assert "vec" in args
        assert "VECTOR" in args
        assert "FLAT" in args

    def test_schema_with_all_field_types(self):
        """A schema with all field types should include them in FT.CREATE."""
        schema = IndexSchema(
            index_name="full_idx",
            prefix="full:",
            vector_fields=[VectorField(name="vec", dimensions=128)],
            text_fields=[TextField(name="title"), TextField(name="content")],
            tag_fields=[TagField(name="category")],
            numeric_fields=[NumericField(name="price", sortable=True)],
        )
        args = schema.build_ft_create_args()

        # Check text fields
        assert "title" in args
        assert "content" in args
        assert "TEXT" in args

        # Check tag fields
        assert "category" in args
        assert "TAG" in args

        # Check numeric fields
        assert "price" in args
        assert "NUMERIC" in args
        assert "SORTABLE" in args

    def test_hnsw_vector_field(self):
        """HNSW fields should include M and EF_CONSTRUCTION params."""
        schema = IndexSchema(
            index_name="hnsw_idx",
            prefix="hnsw:",
            vector_fields=[
                VectorField(
                    name="vec",
                    dimensions=256,
                    algorithm=IndexType.HNSW,
                    m=32,
                    ef_construction=400,
                )
            ],
        )
        args = schema.build_ft_create_args()
        assert "HNSW" in args
        assert "M" in args
        assert "32" in args

    def test_empty_schema_raises_error(self):
        """A schema with no fields should raise SchemaError."""
        schema = IndexSchema(index_name="empty", prefix="empty:")
        with pytest.raises(Exception):
            schema.build_ft_create_args()

    def test_text_field_sortable(self):
        """Sortable text fields should include SORTABLE."""
        schema = IndexSchema(
            index_name="sort_idx",
            prefix="sort:",
            text_fields=[TextField(name="title", sortable=True)],
        )
        args = schema.build_ft_create_args()
        assert "SORTABLE" in args

    def test_tag_field_separator(self):
        """Tag fields should include the custom separator."""
        schema = IndexSchema(
            index_name="tag_idx",
            prefix="tag:",
            tag_fields=[TagField(name="tags", separator=";")],
        )
        args = schema.build_ft_create_args()
        assert "SEPARATOR" in args
        assert ";" in args


class TestIndexManager:
    """Tests for IndexManager (with mocked RediSearch)."""

    async def test_index_exists_returns_false_for_nonexistent(self, conn_manager, mock_redis):
        """index_exists should return False for an index that doesn't exist."""
        from vecstore.core.schema import IndexManager

        mock_redis.set_index_exists(False)
        mgr = IndexManager(conn_manager)
        exists = await mgr.index_exists("nonexistent_index")
        assert exists is False

    async def test_create_and_check_index(self, conn_manager, mock_redis):
        """After creating an index, index_exists should return True."""
        from vecstore.core.schema import IndexManager, VectorField

        mock_redis.set_index_exists(True)
        mgr = IndexManager(conn_manager)
        schema = IndexSchema(
            index_name="test_create_idx",
            prefix="test:create:",
            vector_fields=[VectorField(name="vec", dimensions=4)],
        )

        await mgr.create_index(schema)
        exists = await mgr.index_exists("test_create_idx")
        assert exists is True

    async def test_drop_index(self, conn_manager, mock_redis):
        """After dropping an index, index_exists should return False."""
        from vecstore.core.schema import IndexManager, VectorField

        mgr = IndexManager(conn_manager)
        schema = IndexSchema(
            index_name="test_drop_idx",
            prefix="test:drop:",
            vector_fields=[VectorField(name="vec", dimensions=4)],
        )

        await mgr.create_index(schema)
        await mgr.drop_index("test_drop_idx", delete_docs=True)
        mock_redis.set_index_exists(False)
        exists = await mgr.index_exists("test_drop_idx")
        assert exists is False
