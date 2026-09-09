-- Fixup: register the IVF operator classes in databases where the
-- vectorscale extension predates the ivf access method (the
-- extension_sql! blocks only run at CREATE/ALTER EXTENSION time, so
-- existing installs miss them).  Mirrors the ivf_operator_classes block
-- in pgvectorscale/src/access_method/ivf/mod.rs.
--
-- Usage: psql -v ON_ERROR_STOP=1 -d <db> -f fix_ivf_operator_classes.sql

DO $$
DECLARE
    have_cos_ops int;
    have_l2_ops int;
    have_ip_ops int;
BEGIN
    -- Has cosine operator class been installed for IVF?
    SELECT count(*)
    INTO have_cos_ops
    FROM pg_catalog.pg_opclass c
    WHERE c.opcname = 'vector_cosine_ops'
    AND c.opcmethod = (SELECT oid FROM pg_catalog.pg_am am WHERE am.amname = 'ivf')
    AND c.opcnamespace = (SELECT oid FROM pg_catalog.pg_namespace where nspname='public');

    -- Has L2 operator class been installed for IVF?
    SELECT count(*)
    INTO have_l2_ops
    FROM pg_catalog.pg_opclass c
    WHERE c.opcname = 'vector_l2_ops'
    AND c.opcmethod = (SELECT oid FROM pg_catalog.pg_am am WHERE am.amname = 'ivf')
    AND c.opcnamespace = (SELECT oid FROM pg_catalog.pg_namespace where nspname='public');

    -- Has inner product operator class been installed for IVF?
    SELECT count(*)
    INTO have_ip_ops
    FROM pg_catalog.pg_opclass c
    WHERE c.opcname = 'vector_ip_ops'
    AND c.opcmethod = (SELECT oid FROM pg_catalog.pg_am am WHERE am.amname = 'ivf')
    AND c.opcnamespace = (SELECT oid FROM pg_catalog.pg_namespace where nspname='public');

    IF have_cos_ops = 0 THEN
        CREATE OPERATOR CLASS vector_cosine_ops
        FOR TYPE vector USING ivf AS
            OPERATOR 1 <=> (vector, vector) FOR ORDER BY float_ops,
            FUNCTION 1 distance_type_cosine();
    END IF;

    IF have_l2_ops = 0 THEN
        CREATE OPERATOR CLASS vector_l2_ops
        FOR TYPE vector USING ivf AS
            OPERATOR 1 <-> (vector, vector) FOR ORDER BY float_ops,
            FUNCTION 1 distance_type_l2();
    END IF;

    IF have_ip_ops = 0 THEN
        CREATE OPERATOR CLASS vector_ip_ops
        FOR TYPE vector USING ivf AS
            OPERATOR 1 <#> (vector, vector) FOR ORDER BY float_ops,
            FUNCTION 1 distance_type_inner_product();
    END IF;
END;
$$;
